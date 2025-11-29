# import torchvision.models as models
import torch
import torch.nn as nn
# from graphormer_v1 import Graphormer
from transformers import GraphormerModel

class GraphEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(GraphEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            # LiuC: 第一种导入方式
            # self.graph_model = GraphormerModel.from_pretrained('all_checkpoints/graphormer_pretrained/graphormer-base-pcqm4mv1')
            self.graph_model = GraphormerModel.from_pretrained('all_checkpoints/graphormer_pretrained/graphormer-base-pcqm4mv1')

            # LiuC: 第二种导入方式
            # bertconfig = BertConfig.from_pretrained('../all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/config.json')
            # self.main_model = BertModel(bertconfig)

            # LiuC: 第三种导入方式
            # graph_ckpt_path = '../all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased'
            # self.graph_encoder = Graphormer.load_from_checkpoint(
            #     graph_ckpt_path
            # )

        self.dropout = nn.Dropout(0.1)
        self.graph_encoder = self.graph_model.graph_encoder
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, x, attn_bias, attn_edge_type, spatial_pos,
                                          in_degree, out_degree, edge_input):
        # device = input_ids.device
        device = torch.device(f'cuda:1')
        self.graph_model = self.graph_model.to(device)
        # LiuC: 注意input_nodes
        input_nodes = x
        perturb = None
        # typ = torch.zeros(input_ids.shape).long().to(device)
        # output = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask)['pooler_output']  # b,d
        inner_states, graph_rep = self.graph_encoder(input_nodes, edge_input, attn_bias, in_degree, out_degree,
                                    spatial_pos, attn_edge_type, perturb=perturb)
        # logits = self.dropout(output)
        # return logits
        return graph_rep


if __name__ == '__main__':
    model = GraphEncoder()
    for name, param in model.named_parameters():
        print(name)
