import torch
import torch.nn as nn
import torch.nn.functional as F

# LiuC: TextEncoder GraphEncoder
from model.bert_text import TextEncoder
from model.graphormer import GraphEncoder
from model.MolBT_gt_Miss import MolBT, BT_Config

import pytorch_lightning as pl
from torch import optim

from config import parse_args





# class GraSTSimclr(nn.Module):
class MolBT_CL(pl.LightningModule):
    def __init__(
            self,
            temperature,
            drop_ratio,
            graph_hidden_dim,
            bert_hidden_dim,
            # bert_pretrain,
            projection_dim,
            lr,
            weight_decay,
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

        # Graph Encoder
        # self.graph_encoder = GraphEncoder(pretrained=True)
        self.bt_config = BT_Config()
        self.MolBT_encoder = MolBT(self.bt_config)
        # for name, param in self.graph_encoder.named_parameters():
        #     print(name)

        # Text Encoder
        # self.text_encoder = TextEncoder(pretrained=True)
        # # LiuC: 考虑是否不导入kvplm_pretrained
        # text_ckpt = torch.load('all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
        # if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in text_ckpt:
        #     pretrained_dict = {"main_model." + k[20:]: v for k, v in text_ckpt.items()}
        # elif 'bert.embeddings.word_embeddings.weight' in text_ckpt:
        #     pretrained_dict = {"main_model." + k[5:]: v for k, v in text_ckpt.items()}
        # else:
        #     pretrained_dict = {"main_model." + k[12:]: v for k, v in text_ckpt.items()}
        #
        # self.text_encoder.load_state_dict(pretrained_dict, strict=False)
        # for name, param in self.text_encoder.named_parameters():
        #     print(name)

        # self.feature_extractor.freeze()

        # LiuC: text_proj和graph_proj
        self.graph_proj_head = nn.Sequential(
            nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.graph_hidden_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
            nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )

    def forward(self, features_graph, features_text):

        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss


    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch_idx, batch):
        device = torch.device(f'cuda:1')

        ret = self.MolBT_encoder.infer(batch, graph_token_type_idx=1, irtr_len_text=0)

        loss_rec = ret["loss_rec"]
        text_feature = ret["text_uni_feats"]
        graph_feature = ret["graph_uni_feats"]
        text_cross_feature = ret["text_cross_feats"]
        graph_cross_feature = ret["graph_cross_feats"]



        # LiuC: proj: 768->256
        graph_feature = self.graph_proj_head(graph_feature)
        text_feature = self.text_proj_head(text_feature)
        graph_cross_feature = self.graph_proj_head(graph_cross_feature)
        text_cross_feature = self.text_proj_head(text_cross_feature)

        # LiuC: self.forward()
        _, _, loss1 = self.forward(graph_feature, text_feature)
        _, _, loss2 = self.forward(graph_cross_feature, text_cross_feature)
        _, _, loss3 = self.forward(graph_feature, text_cross_feature)

        loss = (loss1 + loss2 + loss3) / 3.0

        alpha = 5.0
        beta = 1.0
        print("loss:", loss)
        print("loss_rec:", loss_rec)
        loss = alpha * loss_rec + beta * loss

        self.log("train_loss", loss)
        # return loss
        return text_feature, graph_feature, loss


    def training_eval_step(self, batch_idx, batch):
        device = torch.device(f'cuda:1')

        # attn_bias = batch.attn_bias
        # attn_edge_type = batch.attn_edge_type
        # spatial_pos = batch.spatial_pos
        # in_degree = batch.in_degree
        # out_degree = batch.out_degree
        # x = batch.x
        # edge_input = batch.edge_input
        # text
        # text = batch.text
        # text_mask = batch.text_mask
        # text_batch = torch.cat(text, dim=0)
        # text_mask_batch = torch.cat(text_mask, dim=0)

        # graph
        # attn_bias = attn_bias.to(device)
        # attn_edge_type = attn_edge_type.to(device)
        # spatial_pos = spatial_pos.to(device)
        # in_degree = in_degree.to(device)
        # out_degree = out_degree.to(device)
        # x = x.to(device)
        # edge_input = edge_input.to(device)
        # text_batch = text_batch.to(device)
        # text_mask_batch = text_mask_batch.to(device)


        # LiuC: text_feature graph_feature
        # text_feature = self.text_encoder(text_batch, text_mask_batch)
        # graph_feature = self.graph_encoder(x, attn_bias, attn_edge_type, spatial_pos,
        #                               in_degree, out_degree, edge_input)
        ret = self.MolBT_encoder.infer(batch, graph_token_type_idx=1, irtr_len_text=0)
        # graph_feature = ret["graph_cross_feats"]
        text_feature = ret["text_uni_feats"]
        graph_feature = ret["graph_uni_feats"]


        # LiuC: proj: 768->256
        graph_feature = self.graph_proj_head(graph_feature)
        text_feature = self.text_proj_head(text_feature)


        # LiuC: self.forward()
        _, _, loss = self.forward(graph_feature, text_feature)


        self.log("train_loss", loss)
        # return loss
        return text_feature, graph_feature, loss

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("GINSimclr")
    #     # train mode
    #     parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    #
    #     # Graphormer
    #     # parser.add_argument('--gin_hidden_dim', type=int, default=300)
    #     # parser.add_argument('--gin_num_layers', type=int, default=5)
    #     # parser.add_argument('--drop_ratio', type=float, default=0.0)
    #     # parser.add_argument('--graph_pooling', type=str, default='sum')
    #
    #     # Bert
    #     parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    #     # parser.add_argument('--bert_pretrain', action='store_false', default=True)
    #     parser.add_argument('--projection_dim', type=int, default=256)
    #     # optimization
    #     parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    #     parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
    #     return parent_parser


if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device(f'cuda:0')

    graph_encoder = GraphEncoder(pretrained=True)
    for name, param in graph_encoder.named_parameters():
        print(name)

    print('##############')
    text_encoder = TextEncoder(pretrained=True)
    text_ckpt = torch.load('../all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
    if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in text_ckpt:
        pretrained_dict = {"main_model." + k[20:]: v for k, v in text_ckpt.items()}
    elif 'bert.embeddings.word_embeddings.weight' in text_ckpt:
        pretrained_dict = {"main_model." + k[5:]: v for k, v in text_ckpt.items()}
    else:
        pretrained_dict = {"main_model." + k[12:]: v for k, v in text_ckpt.items()}

    text_encoder.load_state_dict(pretrained_dict, strict=False)
    for name, param in text_encoder.named_parameters():
        print(name)