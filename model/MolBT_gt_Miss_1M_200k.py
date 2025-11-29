import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from model.bert_model import BertCrossLayer

from transformers import BertModel, BertConfig
from transformers import GraphormerModel
# from . import heads, objectives, meter_utils
from model import heads, objectives
from transformers import RobertaConfig, RobertaModel

from model.miss_module.glow import Glow, ZeroConv2d, gaussian_log_p
from model.miss_module.rcan import Group

# LiuC:
import os
import re
import numpy as np
from config import parse_args
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from data_provider.GraST_dataset_gts import GraSTPretrain_STM
from data_provider.wrapper import preprocess_item
from data_provider.collator_gts import collator, Batch
from functools import partial


class BT_Config:
    def __init__(self):
        # PL Trainer Setting
        self.resume_from = None
        self.fast_dev_run = False
        self.val_check_interval = 1.0
        self.text_only = False
        self.log_every_n_steps = 50

        # Experiment Setting
        self.seed = 42
        self.batch_size = 16

        # text Setting
        self.vqav2_label_size = 3129
        self.max_text_len = 128
        self.vocab_size = 30522
        self.whole_word_masking = False
        self.mlm_prob = 0.15
        self.draw_false_text = 0

        # Transformer Setting
        self.input_graph_embed_size = 768
        self.input_text_embed_size = 768
        self.hidden_size = 768
        self.num_heads = 12
        self.num_layers = 6
        self.mlp_ratio = 4
        self.drop_rate = 0.1

        # Optimizer Setting
        self.optim_type = "adamw"
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.decay_power = 1
        self.max_epoch = 10
        self.max_steps = -1
        self.warmup_steps = 10000
        self.end_lr = 0
        self.lr_mult_head = 5
        self.lr_mult_cross_modal = 5

        # BT Setting
        self.model_type = "BT"
        self.vit_layernorm_shared = True
        self.vit_layernorm_init_from_vit = False
        self.task_head_layers = 2
        self.head_hidden_scale = 1
        self.per_gpu_eval_batchsize_text = 256
        self.per_gpu_eval_batchsize_graph = 128
        self.per_gpu_eval_batchsize_fusion_text = 500
        self.k_test = 128
        self.amp_flag = True
        self.task_threshold = 0
        self.nlvr2_drop_rate = 0.1

        # Contrastive Setting
        self.temperature = 0.07
        self.contrastive_hidden_size = 256
        self.gather_with_grads = True
        self.gather_global_negative = False
        self.gather_all_graph_inputs = False
        self.graph_chunks = 1
        self.text_chunks = 1
        self.save_memory = False

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse



class MolBT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )


        self.cross_modal_text_transform = nn.Linear(config.input_text_embed_size, config.hidden_size)
        self.cross_modal_graph_transform = nn.Linear(config.input_graph_embed_size, config.hidden_size)
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_graph_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        self.text_transformer = BertModel.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased')
        text_ckpt = torch.load('all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in text_ckpt:
            pretrained_dict = {"main_model." + k[20:]: v for k, v in text_ckpt.items()}
        elif 'bert.embeddings.word_embeddings.weight' in text_ckpt:
            pretrained_dict = {"main_model." + k[5:]: v for k, v in text_ckpt.items()}
        else:
            pretrained_dict = {"main_model." + k[12:]: v for k, v in text_ckpt.items()}
        self.text_transformer.load_state_dict(pretrained_dict, strict=False)

        self.graph_model = GraphormerModel.from_pretrained(
            'all_checkpoints/graphormer_pretrained/graphormer-base-pcqm4mv1')
        self.graph_encoder = self.graph_model.graph_encoder
        self.graph_node_feature = self.graph_encoder.graph_node_feature
        self.graph_attn_bias = self.graph_encoder.graph_attn_bias

        self.cross_modal_graph_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config.num_layers)])
        self.cross_modal_graph_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config.num_layers)])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        # Class token => Linear => Tanh
        self.cross_modal_graph_pooler = heads.Pooler(config.hidden_size)
        self.cross_modal_graph_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config.hidden_size)
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        # Temperature for graph text contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * config.temperature)


        hs = config.hidden_size

        # ===================== Initialize BT Components ===================== #
        # just for first layer
        self.cross_modal_text_layernorm = nn.LayerNorm(config.hidden_size)
        self.cross_modal_text_layernorm.apply(objectives.init_weights)
        self.cross_modal_graph_layernorm = nn.LayerNorm(config.hidden_size)
        self.cross_modal_graph_layernorm.apply(objectives.init_weights)


        self.cross_modal_text_link_tower = nn.ModuleList(
            [heads.LinkTower(config) for _ in range(config.num_layers - 1)])
        self.cross_modal_graph_link_tower = nn.ModuleList(
            [heads.LinkTower(config) for _ in range(config.num_layers - 1)])

        self.cross_modal_text_link_tower.apply(objectives.init_weights)
        self.cross_modal_graph_link_tower.apply(objectives.init_weights)


        # LiuC:
        self.MSE = MSE()

        dst_feature_dims = 64
        nheads = 8
        self.orig_d_t = self.orig_d_s = self.orig_d_g = 768
        self.d_t = self.d_s = self.d_g = dst_feature_dims
        self.num_heads = nheads

        # LiuC: Miss Module
        self.flow_t = Glow(in_channel=self.d_t, n_flow=32, n_block=1, affine=True, conv_lu=False)
        self.flow_s = Glow(in_channel=self.d_s, n_flow=32, n_block=1, affine=True, conv_lu=False)
        self.flow_g = Glow(in_channel=self.d_g, n_flow=32, n_block=1, affine=True, conv_lu=False)

        self.rec_t = nn.Sequential(
            nn.Conv1d(self.d_t, self.d_t * 2, 1),
            Group(num_channels=self.d_t * 2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_t * 2, self.d_t, 1)
        )

        self.cat_t = nn.Conv1d(self.d_t * 2, self.d_t, kernel_size=1, padding=0)

        # 1. Temporal convolutional layers
        self.proj_t = nn.Conv1d(self.orig_d_t, self.d_t, kernel_size=5, padding=0, bias=False)
        self.proj_s = nn.Conv1d(self.orig_d_s, self.d_s, kernel_size=5, padding=0, bias=False)
        self.proj_g = nn.Conv1d(self.orig_d_g, self.d_g, kernel_size=5, padding=0, bias=False)

        # LiuC: 64 -> 768
        self.linear_layer = nn.Linear(64, 768)


    # LiuC:
    def infer(
            self,
            batch,
            # mask_text=False,
            # mask_graph=False,
            graph_token_type_idx=1,
            # img=None,
            irtr_len_text=0,
    ):
        device = torch.device(f'cuda:1')


        # text_ids = batch[f"text_ids"]
        text_ids = batch.text
        # text_labels = batch[f"text_labels"]
        # text_masks = batch[f"text_masks"]
        text_masks = batch.text_mask
        text_ids = torch.cat(text_ids, dim=0)
        text_masks = torch.cat(text_masks, dim=0)

        text_ids = text_ids.to(device)
        text_masks = text_masks.to(device)

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, self.device)

        split_index = len(self.text_transformer.encoder.layer) - self.hparams.config.num_layers + 1
        for layer in self.text_transformer.encoder.layer[:split_index]:
            text_embeds = layer(text_embeds, extend_text_masks)[0]


        # smiles
        smiles_ids = batch.smiles
        smiles_masks = batch.smiles_mask
        smiles_ids = torch.cat(smiles_ids, dim=0)
        smiles_masks = torch.cat(smiles_masks, dim=0)
        smiles_ids = smiles_ids.to(device)
        smiles_masks = smiles_masks.to(device)

        smiles_embeds = self.text_transformer.embeddings(input_ids=smiles_ids)
        smiles_input_shape = smiles_masks.size()
        extend_smiles_masks = self.text_transformer.get_extended_attention_mask(smiles_masks, smiles_input_shape, self.device)

        split_index = len(self.text_transformer.encoder.layer) - self.hparams.config.num_layers + 1
        for layer in self.text_transformer.encoder.layer[:split_index]:
            smiles_embeds = layer(smiles_embeds, extend_smiles_masks)[0]

        # if self.is_clip:
        #     image_embeds = self.vit_model.visual.forward_pre(img.type(self.vit_model.dtype))
        #     for block in self.vit_model.visual.transformer.resblocks[:split_index]:
        #         image_embeds = block(image_embeds)
        #     image_embeds_ = self.vit_model.visual.forward_post(image_embeds.type(self.vit_model.dtype))
        # else:
        #     image_embeds = self.vit_model.forward_pre(img)
        #     for block in self.vit_model.blocks[:split_index]:
        #         image_embeds = block(image_embeds)
        #     image_embeds_ = self.vit_model.forward_post(image_embeds)

        # graph
        attn_bias = batch.attn_bias
        attn_edge_type = batch.attn_edge_type
        spatial_pos = batch.spatial_pos
        in_degree = batch.in_degree
        out_degree = batch.out_degree
        x = batch.x
        edge_input = batch.edge_input

        attn_bias = attn_bias.to(device)
        attn_edge_type = attn_edge_type.to(device)
        spatial_pos = spatial_pos.to(device)
        in_degree = in_degree.to(device)
        out_degree = out_degree.to(device)
        x = x.to(device)
        edge_input = edge_input.to(device)

        input_nodes = x
        perturb = None
        inner_states, graph_rep = self.graph_encoder(input_nodes, edge_input, attn_bias, in_degree, out_degree,
                                                     spatial_pos, attn_edge_type, perturb=perturb)
        data_x = input_nodes
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, edge_input, attn_edge_type)

        inner = inner_states[0]
        inner_re = inner.clone()

        inner_list = []

        for layer in self.graph_model.graph_encoder.layers[:split_index]:
            inner_re, _ = layer(
                input_nodes=inner_re,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=None,
                self_attn_bias=attn_bias
            )
            inner_list.append(inner_re)
        # graph_embeds_ = inner_re[0, :, :]
        graph_embeds_ = inner_re.transpose(0, 1)
        # graph_embeds_ = inner_re

        # # LiuC:
        # x_t = text_embeds.transpose(1, 2)
        # x_s = smiles_embeds.transpose(1, 2)
        # x_g = graph_embeds_.transpose(1, 2)
        #
        # with torch.no_grad():
        #     proj_x_t = x_t if self.orig_d_t == self.d_t else self.proj_t(x_t)
        #     proj_x_s = x_s if self.orig_d_s == self.d_s else self.proj_s(x_s)
        #     proj_x_g = x_g if self.orig_d_g == self.d_g else self.proj_g(x_g)
        #
        # conv_feat_t, conv_feat_s, conv_feat_g = proj_x_t, proj_x_s, proj_x_g
        #
        # #  normalizing flow for language
        # # _, logdet_t, z_outs_t = self.flow_t(proj_x_t.unsqueeze(-1))
        # # z_t = z_outs_t
        # # z_outs_t = z_outs_t[0]
        #
        # #  normalizing flow for vision
        # _, logdet_s, z_outs_s = self.flow_s(proj_x_s.unsqueeze(-1))
        # z_s = z_outs_s
        # z_outs_s = z_outs_s[0]
        #
        # #  normalizing flow for audio
        # _, logdet_g, z_outs_g = self.flow_g(proj_x_g.unsqueeze(-1))
        # z_g = z_outs_g
        # z_outs_g = z_outs_g[0]
        #
        # # Generate Text:  (g,s) to t
        # t_1 = self.flow_t.reverse(z_g, reconstruct=True).squeeze(-1).detach()
        # des_size = 60
        # current_size = t_1.size(2)  # 获取 t_1 在 dim=2 上的大小
        # if current_size > des_size:
        #     # 如果当前大小大于 64，裁剪到 64
        #     t_1 = t_1[:, :, :des_size]
        # elif current_size < des_size:
        #     # 如果当前大小小于 64，插值到 64
        #     t_1 = F.interpolate(t_1, size=des_size, mode='linear', align_corners=False)
        #
        # t_2 = self.flow_t.reverse(z_s, reconstruct=True).squeeze(-1).detach()
        # proj_x_t = self.cat_t(torch.cat([t_1, t_2], dim=1))
        # proj_x_t = self.rec_t(proj_x_t)
        #
        # proj_x_t = F.interpolate(proj_x_t, size=128, mode='linear', align_corners=False)
        # proj_x_t = proj_x_t.transpose(1, 2)
        # proj_x_t = self.linear_layer(proj_x_t)
        #
        # # conv_feat_t = F.interpolate(conv_feat_t.detach(), size=128, mode='linear', align_corners=False)
        # # loss_rec = self.MSE(proj_x_t, conv_feat_t.detach())
        # loss_rec = self.MSE(proj_x_t, text_embeds.detach())
        #
        # proj_x_t_masks = (proj_x_t.sum(dim=-1) != 0).long()
        # input_shape = proj_x_t_masks.size()
        # proj_x_t_masks = proj_x_t_masks.to(device)
        # extend_proj_x_t_masks = self.text_transformer.get_extended_attention_mask(proj_x_t_masks, input_shape, self.device)


        # first layer
        # text_embeds = proj_x_t
        # extend_text_masks = extend_proj_x_t_masks

        x = self.cross_modal_text_transform(text_embeds)
        text_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device)).expand_as(x)
        x = self.cross_modal_text_layernorm(x + text_token_type_embeddings)

        graph_embeds_ = self.cross_modal_graph_transform(graph_embeds_)
        graph_token_type_embeddings = self.token_type_embeddings(
            torch.zeros(1).long().to(self.device).fill_(graph_token_type_idx)).expand_as(graph_embeds_)
        graph_embeds_ = graph_embeds_ + graph_token_type_embeddings
        y = self.cross_modal_graph_layernorm(graph_embeds_)
        # if irtr_len_text > 0:
        #     _bs, _L, _D = graph_embeds_.size()
        #     y = y.unsqueeze(1).expand(_bs, irtr_len_text, _L, _D).contiguous().view(-1, _L, _D)
        graph_masks = torch.ones((y.size(0), y.size(1)), dtype=torch.long, device=self.device)
        extend_graph_masks = self.text_transformer.get_extended_attention_mask(graph_masks, graph_masks.size(),
                                                                               self.device)

        # 从smiles_embeds中提取cls token
        smiles_cls = smiles_embeds[:,0]

        # 将smiles_cls添加到cross Attention
        x1 = x.clone()
        y1 = y.clone()
        x1[:, -1, :] = smiles_cls
        y1[:, -1, :] = smiles_cls
        x1 = self.cross_modal_text_layers[0](x, y1, extend_text_masks, extend_graph_masks)[0]
        y1 = self.cross_modal_graph_layers[0](y, x1, extend_graph_masks, extend_text_masks)[0]

        link_layer_index = 0

        # link tower fusion
        for i in range(split_index, len(self.text_transformer.encoder.layer)):
            text_embeds_uni = self.text_transformer.encoder.layer[i](text_embeds, extend_text_masks)[0]
            smiles_embeds_uni = self.text_transformer.encoder.layer[i](smiles_embeds, extend_smiles_masks)[0]
            smiles_cls = smiles_embeds_uni[:, 0]
            # if self.is_clip:
            #     image_embeds = self.vit_model.visual.transformer.resblocks[i](image_embeds).type(self.vit_model.dtype)
            #     image_embeds_ = self.cross_modal_image_transform(
            #         self.vit_model.visual.forward_post(image_embeds)) + image_token_type_embeddings
            # else:
            #     image_embeds = self.vit_model.blocks[i](image_embeds)
            #     image_embeds_ = self.cross_modal_image_transform(
            #         self.vit_model.forward_post(image_embeds)) + image_token_type_embeddings
            graph_embeds_ = graph_embeds_.transpose(0, 1)
            graph_embeds_uni, _ = self.graph_model.graph_encoder.layers[i](input_nodes=graph_embeds_,
                                                                    self_attn_padding_mask=padding_mask,
                                                                    self_attn_mask=None,
                                                                    self_attn_bias=attn_bias)
            graph_embeds_ = graph_embeds_uni.transpose(0, 1)
            graph_embeds_ = self.cross_modal_graph_layernorm(graph_embeds_) + graph_token_type_embeddings

            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            graph_link_tower = self.cross_modal_graph_link_tower[link_layer_index]

            x1_ = text_link_tower(self.cross_modal_text_transform(text_embeds_uni) + text_token_type_embeddings, x1)
            # if irtr_len_text > 0:
            #     y1_ = image_link_tower(
            #         image_embeds_.unsqueeze(1).expand(_bs, irtr_len_text, _L, _D).contiguous().view(-1, _L, _D), y1)
            # else:
            #     y1_ = image_link_tower(image_embeds_, y1)
            y1_ = graph_link_tower(graph_embeds_, y1)

            # 将smiles添加到cross Attention中
            x1_1 = x1_.clone()
            y1_1 = y1_.clone()
            x1_1[:, -1, :] = smiles_cls
            y1_1[:, -1, :] = smiles_cls

            x1 = self.cross_modal_text_layers[link_layer_index + 1](x1_, y1_1, extend_text_masks, extend_graph_masks)[0]
            y1 = self.cross_modal_graph_layers[link_layer_index + 1](y1_, x1_1, extend_graph_masks, extend_text_masks)[0]

            link_layer_index += 1

        text_feats, graph_feats = x1, y1
        # cls_feats = self.get_cls_feats(text_feats, graph_feats)

        text_feats = text_feats[:, 0, :]
        graph_feats = graph_feats[:, 0, :]
        text_embeds_uni = text_embeds_uni[:, 0, :]
        graph_embeds_uni = graph_embeds_uni[0, :, :]

        ret = {
            # "loss_rec": loss_rec,
            "text_cross_feats": text_feats,
            "graph_cross_feats": graph_feats,
            "text_uni_feats": text_embeds_uni,
            "graph_uni_feats": graph_embeds_uni,
        }

        return ret




    # def forward(self, batch, split):
    #     ret = dict()
    #     if len(self.current_tasks) == 0:
    #         ret.update(self.infer(batch))
    #         return ret
    #
    #     # Masked Language Modeling
    #     if "mlm" in self.current_tasks:
    #         ret.update(objectives.compute_mlm(self, batch, split))
    #
    #     # Image Text Matching
    #     if "itm" in self.current_tasks:
    #         ret.update(objectives.compute_itm(self, batch, split))
    #
    #     if "itc" in self.current_tasks:
    #         ret.update(objectives.compute_itc(self, batch, split))

        # if "itm_itc" in self.current_tasks:
        #     ret.update(objectives.compute_itm_itc(self, batch, split, pretrain=True))
        #
        # if "irtr_itm_itc" in self.current_tasks:
        #     ret.update(objectives.compute_itm_itc(self, batch, split, pretrain=False))

        # # Visual Question Answering
        # if "vqa" in self.current_tasks:
        #     ret.update(objectives.compute_vqa(self, batch, split))
        #
        # # Natural Language for Visual Reasoning 2
        # if "nlvr2" in self.current_tasks:
        #     ret.update(objectives.compute_nlvr2(self, batch, split))
        #
        # # SNLI Visual Entailment
        # if "snli" in self.current_tasks:
        #     ret.update(objectives.compute_snli(self, batch, split))
        #
        # # Image Retrieval and Text Retrieval
        # if "irtr" in self.current_tasks:
        #     ret.update(objectives.compute_irtr(self, batch, split))

        # return ret


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # LiuC:
    config = BT_Config()
    device = torch.device("cuda:1")

    # LiuC: 通过text文件名中的序号 获取ids
    ids = []
    # text_name_list = os.listdir("../data/kv_data/text_15k")
    text_name_list = os.listdir("../preprocessing/PubChemSTM/PubChemSTM_process/text_PubChemSTM")
    for text_name in text_name_list:
        text_id = re.split('[_.]', text_name)[1]
        text_id = int(text_id)
        ids.append(text_id)
    ids.sort()
    # print(ids) [1, 2, 3, 4, 5, 6, 7, ..., 14999, 15000]
    seq = np.arange(len(ids))
    print(seq)
    print(len(seq))  # 14992
    np.random.shuffle(seq)
    print(seq)
    print(len(seq))  # 14992

    scaf = []
    k = int(len(seq) / 10)
    scaf.append(seq[:7 * k])
    scaf.append(seq[7 * k:8 * k])
    scaf.append(seq[8 * k:])

    # TrainSet = GraSTMatchDataset(args, ids, scaf[0])
    # DevSet = GraSTMatchDataset(args, ids, scaf[1])
    # TestSet = GraSTMatchDataset(args, ids, scaf[2])
    TrainSet = GraSTPretrain_STM(args, ids, scaf[0])
    DevSet = GraSTPretrain_STM(args, ids, scaf[1])
    TestSet = GraSTPretrain_STM(args, ids, scaf[2])
    train_sampler = RandomSampler(TrainSet)

    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=True,
                                  collate_fn=partial(collator, max_node=128,
                                                     multi_hop_max_dist=20, spatial_pos_max=20))
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=0, pin_memory=True, drop_last=True,
                                collate_fn=partial(collator, max_node=128,
                                                   multi_hop_max_dist=20, spatial_pos_max=20))
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=0, pin_memory=True, drop_last=True,
                                 collate_fn=partial(collator, max_node=128,
                                                    multi_hop_max_dist=20, spatial_pos_max=20))

    for epoch in range(args.epoch):
        print(args.epoch)
        for idx, batch in enumerate((train_dataloader)):
            # batch = batch.to(device)
            # # print(f"Batch {idx}: {batch}")
            #
            # # graph
            # # idx = batch.idx
            # attn_bias = batch.attn_bias
            # attn_edge_type = batch.attn_edge_type
            # spatial_pos = batch.spatial_pos
            # in_degree = batch.in_degree
            # out_degree = batch.out_degree
            # x = batch.x
            # edge_input = batch.edge_input
            # # text
            # text = batch.text
            # text_mask = batch.text_mask
            # text_batch = torch.cat(text, dim=0)
            # text_mask_batch = torch.cat(text_mask, dim=0)
            #
            # # batch_idx,attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input = batch
            #
            # # LiuC: process  text tuple
            #
            # # LiuC: solve: data = data.to(device)
            # attn_bias = attn_bias.to(device)
            # attn_edge_type = attn_edge_type.to(device)
            # spatial_pos = spatial_pos.to(device)
            # in_degree = in_degree.to(device)
            # out_degree = out_degree.to(device)
            # x = x.to(device)
            # edge_input = edge_input.to(device)
            #
            # text_batch = text_batch.to(device)
            # text_mask_batch = text_mask_batch.to(device)

            ret = dict()
            model = MolBT(config)
            model.to(device)
            ret = model.infer(batch, graph_token_type_idx=1, irtr_len_text=0)
            print("hhh")






