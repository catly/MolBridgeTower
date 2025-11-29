import torch
import torch.nn as nn
import torch.nn.functional as F


from model.bert_model import BertCrossLayer
from transformers import BertModel, BertConfig
from transformers import GraphormerModel
from model import heads, objectives

import pytorch_lightning as pl
from torch import optim

# from config import parse_args

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


class Graph_pred(pl.LightningModule):
    def __init__(self, config,
                 temperature,
                 drop_ratio,
                 graph_hidden_dim,
                 bert_hidden_dim,
                 # bert_pretrain,
                 projection_dim,
                 lr,
                 weight_decay,
                 num_tasks
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        self.drop_ratio = drop_ratio
        self.graph_hidden_dim = graph_hidden_dim
        self.bert_hidden_dim = bert_hidden_dim
        # self.bert_pretrain = bert_pretrain
        self.projection_dim = projection_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_tasks = num_tasks

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

        self.text_transformer = BertModel.from_pretrained(
            'all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased')
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

        # For graph-level binary classification
        self.mult = 1
        self.graph_pred_linear = nn.Linear(self.mult * self.graph_hidden_dim, self.num_tasks)


    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def forward(self, batch):
        device = torch.device(f'cuda:1')
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

        for layer in self.graph_model.graph_encoder.layers:
            inner_re, _ = layer(
                input_nodes=inner_re,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=None,
                self_attn_bias=attn_bias
            )
            inner_list.append(inner_re)
        graph_embeds = inner_re[0, :, :]
        # graph_embeds = inner_re.transpose(0, 1)
        # graph_embeds_ = inner_re


        output = self.graph_pred_linear(graph_embeds)

        # return loss
        # return text_feature, graph_feature, loss
        return graph_embeds, output
