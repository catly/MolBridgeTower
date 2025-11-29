# import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import argparse
from config import parse_args

class TextEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            # LiuC: 第一种导入方式
            # self.main_model = BertModel.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased') \
            self.main_model = BertModel.from_pretrained(
                        '../all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased')

            # LiuC: 第二种导入方式
            # bertconfig = BertConfig.from_pretrained('../all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/config.json')
            # self.main_model = BertModel(bertconfig)

        self.embeddings = self.main_model.embeddings
        self.encoder = self.main_model.encoder
        self.pooler = self.main_model.pooler

        self.dropout = nn.Dropout(0.1)

        # self.hidden_size = self.main_model.config.hidden_size

    # def forward(self, input_ids, attention_mask, prompt):
    def forward(self, input_ids, attention_mask, prompt_emb, prompt_extended_mask):
        # device = input_ids.device
        device = torch.device(f'cuda:1')
        # pooler
        # self.dense = nn.Linear(768, 768)
        # self.activation = nn.Tanh()

        self.main_model = self.main_model.to(device)
        typ = torch.zeros(input_ids.shape).long().to(device)

        hidden_states_0 = self.embeddings(input_ids, token_type_ids=typ)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        text_extended_mask = (1.0 - extended_attention_mask) * -10000.0

        processed_emb = torch.cat((hidden_states_0, prompt_emb), dim=1)
        extended_attention_mask = torch.cat((text_extended_mask, prompt_extended_mask), dim=-1)

        output = self.encoder(processed_emb, extended_attention_mask)[0]
        output = self.pooler(output)


        # output_1 = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask,
        #                            output_attentions=True, output_hidden_states=True)
        # output_2 = output_1['pooler_output']  # b,d

        # # Bert的12层encoder
        # encoder_layers = self.main_model.encoder.layer  # 12层
        # # hs_list：每层输入与最终输出
        # hs_list = []
        # hidden_states = output_1.hidden_states # len:13
        # hs_list[0] = hidden_states[0] # hs_list.append()
        #
        # process_input = []

        # for i in range(12):
        #     # 处理 prompt与各层输出，得到各层输入
        #
        #     hs = encoder_layers[i](process_input[i])
        #     hs_list[i] = hs[0]  # hs[0]是tensor
        #     # hs_list.append()


        # # pooler
        # first_token = hidden_states[:, 0]
        # output = self.dense(first_token)
        # output = self.activation(output)

        # logits = self.dropout(output)
        # return logits
        return output


if __name__ == '__main__':
    model = TextEncoder()
    for name, param in model.named_parameters():
        print(name)
    # print(model.modules())
    # print(model.modules())
    # x0 = [0.1,0.2,0.3]
    # for layer in model.modules():
    #     x = layer(x0)
    #     print(x)
