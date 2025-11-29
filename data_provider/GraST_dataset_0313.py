import torch
# from torch_geometric.data import Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaModel

# LiuC: test if __name__ == '__main__':
from config import parse_args
import re
from torch.utils.data import RandomSampler
import torch_geometric

# LiuC:
from data_provider.wrapper import preprocess_item
from data_provider.collator_0313 import collator, Batch

# LiuC: graph Encoder and text Encoder
# from model.contrastive_GraST import GraSTSimclr
import torch.nn as nn

# LiuC: Graphormer
# from transformers import GraphormerModel

# from model.graphormer import GraphEncoder
# from model.bert import TextEncoder

# from Test_Code.test_0304_graph_encoder import GraphEncoder
# from Test_Code.test_0304_text_encoder import TextEncoder

from model.graphormer_0313 import GraphEncoder
from model.bert_0313 import TextEncoder
from model.prompt_encoder import PromptEncoder

import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
# LiuC:
from functools import partial


class GraSTMatchDataset(Dataset):
    def __init__(self, args, ids, scaf):
        super(GraSTMatchDataset, self).__init__()
        self.ids = ids
        self.scaf = scaf
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.prompt_max_len = args.prompt_max_len
        self.tokenizer = BertTokenizer.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/')
        # self.data_type = args.data_type

    def __len__(self):
        return len(self.scaf)

    def __getitem__(self, index):
        # graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        idx = self.scaf[index]
        # LiuC: data/kv_data/{mode}/graph'
        # graph_path = '../data/kv_data/graph/graph' + '_' + f"{self.ids[idx]}.pt"
        # text_path = '../data/kv_data/text/text' + '_' + f"{self.ids[idx]}.txt"
        graph_path = 'data/kv_data/graph_15k/graph' + '_' + f"{self.ids[idx]}.pt"
        text_path = 'data/kv_data/text_15k/text' + '_' + f"{self.ids[idx]}.txt"
        prompt_path = 'data/kv_data/prompt_15k/prompt' + '_' + f"{self.ids[idx]}.txt"
        # smile_path = '../data/kv_data/smiles_15k/smiles' + '_' + f"{self.ids[idx]}.txt"

        # load graph data
        graph = torch.load(graph_path)
        # LiuC: process data_graph

        # if isinstance(idx, int):
        #     graph = self.get(self.indices()[idx])
        #     graph.idx = idx
        data_graph_text = preprocess_item(graph)
        data_graph_text.idx = idx

        # print(data_graph) # contain a lot of attributes

        # load text data
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 1:
                break

        # load prompt_text data
        prompt_text_list = []
        count_prompt = 0
        for line in open(prompt_path, 'r', encoding='utf-8'):
            count_prompt += 1
            line.strip('\n')
            prompt_text_list.append(line)
            if count > 1:
                break

        # load smiles data
        # smiles_list = []
        # count = 0
        # for line in open(smile_path, 'r', encoding='utf-8'):
        #     count += 1
        #     line.strip('\n')
        #     smiles_list.append(line)
        #     if count > 1:
        #         break

        # smiles+text  待写
        smilestext_list = []

        text = mask = None
        prompt_text = prompt_mask = None

        # if self.data_type == 'para':  # paragraph-level
        text, mask = self.tokenizer_text(text_list[0])
        prompt_text, prompt_mask = self.tokenizer_prompt(prompt_text_list[0])

        # LiuC:
        data_graph_text.text = text
        data_graph_text.text_mask = mask
        data_graph_text.prompt_text = prompt_text
        data_graph_text.prompt_mask = prompt_mask


        # return data_graph, text.squeeze(0), mask.squeeze(0)  # , index
        return data_graph_text
        # return graph, text.squeeze(0), mask.squeeze(0)  # , index
        # return text.squeeze(0), mask.squeeze(0)


    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

    def tokenizer_prompt(self, prompt_text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=prompt_text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.prompt_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


class GraSTMatchDataset_MoMu(Dataset):
    def __init__(self, args, ids, scaf):
        super(GraSTMatchDataset_MoMu, self).__init__()
        self.ids = ids
        self.scaf = scaf
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.prompt_max_len = args.prompt_max_len
        self.tokenizer = BertTokenizer.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/')
        # self.data_type = args.data_type

    def __len__(self):
        return len(self.scaf)

    def __getitem__(self, index):
        # graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        idx = self.scaf[index]
        # LiuC: data/kv_data/{mode}/graph'
        # graph_path = '../data/kv_data/graph/graph' + '_' + f"{self.ids[idx]}.pt"
        # text_path = '../data/kv_data/text/text' + '_' + f"{self.ids[idx]}.txt"
        graph_path = 'data/MoMu_data/graph/graph' + '_' + f"{self.ids[idx]}.pt"
        text_path = 'data/MoMu_data/text/text' + '_' + f"{self.ids[idx]}.txt"
        prompt_path = 'data/MoMu_data/prompt/prompt' + '_' + f"{self.ids[idx]}.txt"
        # smile_path = '../data/kv_data/smiles_15k/smiles' + '_' + f"{self.ids[idx]}.txt"

        # load graph data
        graph = torch.load(graph_path)
        # LiuC: process data_graph

        # if isinstance(idx, int):
        #     graph = self.get(self.indices()[idx])
        #     graph.idx = idx
        data_graph_text = preprocess_item(graph)
        data_graph_text.idx = idx

        # print(data_graph) # contain a lot of attributes

        # load text data
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 1:
                break

        # load prompt_text data
        prompt_text_list = []
        count_prompt = 0
        for line in open(prompt_path, 'r', encoding='utf-8'):
            count_prompt += 1
            line.strip('\n')
            prompt_text_list.append(line)
            if count > 1:
                break

        # load smiles data
        # smiles_list = []
        # count = 0
        # for line in open(smile_path, 'r', encoding='utf-8'):
        #     count += 1
        #     line.strip('\n')
        #     smiles_list.append(line)
        #     if count > 1:
        #         break

        # smiles+text  待写
        smilestext_list = []

        text = mask = None
        prompt_text = prompt_mask = None

        # if self.data_type == 'para':  # paragraph-level
        text, mask = self.tokenizer_text(text_list[0])
        prompt_text, prompt_mask = self.tokenizer_prompt(prompt_text_list[0])

        # LiuC:
        data_graph_text.text = text
        data_graph_text.text_mask = mask
        data_graph_text.prompt_text = prompt_text
        data_graph_text.prompt_mask = prompt_mask


        # return data_graph, text.squeeze(0), mask.squeeze(0)  # , index
        return data_graph_text
        # return graph, text.squeeze(0), mask.squeeze(0)  # , index
        # return text.squeeze(0), mask.squeeze(0)


    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

    def tokenizer_prompt(self, prompt_text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=prompt_text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.prompt_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


# class MyGraSTDataset(GraSTMatchDataset):
#     def __getitem__(self, idx):
#         if isinstance(idx, int):
#             item = self.get(self.indices()[idx])
#             item.idx = idx
#             return preprocess_item(item)
#         else:
#             return self.index_select(idx)


class GraSTPretrain_STM(Dataset):
    def __init__(self, args, ids, scaf):
        # super(GraSTMatchDataset, self).__init__()
        super(GraSTPretrain_STM, self).__init__()
        self.ids = ids
        self.scaf = scaf
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.prompt_max_len = args.prompt_max_len
        self.tokenizer = BertTokenizer.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/')
        # self.data_type = args.data_type

    def __len__(self):
        return len(self.scaf)

    def __getitem__(self, index):
        # graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        idx = self.scaf[index]
        # LiuC: data/kv_data/{mode}/graph'
        # graph_path = './data/kv_data/graph/graph' + '_' + f"{self.ids[idx]}.pt"
        # text_path = './data/kv_data/text/text' + '_' + f"{self.ids[idx]}.txt"
        graph_path = 'preprocessing/PubChemSTM/PubChemSTM_process/graph_PubChemSTM/graph' + '_' + f"{self.ids[idx]}.pt"
        text_path = 'preprocessing/PubChemSTM/PubChemSTM_process/text_PubChemSTM/text' + '_' + f"{self.ids[idx]}.txt"
        prompt_path = 'preprocessing/PubChemSTM/PubChemSTM_process/prompt_PubChemSTM/prompt' + '_' + f"{self.ids[idx]}.txt"
        # smile_path = '../data/kv_data/smiles_15k/smiles' + '_' + f"{self.ids[idx]}.txt"
        # LiuC: 0305
        smile_path = 'preprocessing/PubChemSTM/PubChemSTM_process/smiles_PubChemSTM/smiles' + '_' + f"{self.ids[idx]}.txt"

        # load graph data
        graph = torch.load(graph_path)
        # LiuC: process data_graph

        # if isinstance(idx, int):
        #     graph = self.get(self.indices()[idx])
        #     graph.idx = idx
        data_graph_text = preprocess_item(graph)
        data_graph_text.idx = idx

        # print(data_graph) # contain a lot of attributes

        # load text data
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 1:
                break

        # load prompt_text data
        prompt_text_list = []
        count_prompt = 0
        for line in open(prompt_path, 'r', encoding='utf-8'):
            count_prompt += 1
            line.strip('\n')
            prompt_text_list.append(line)
            if count > 1:
                break

        # load smiles data
        # smiles_list = []
        # count = 0
        # for line in open(smile_path, 'r', encoding='utf-8'):
        #     count += 1
        #     line.strip('\n')
        #     smiles_list.append(line)
        #     if count > 1:
        #         break

        # smiles+text  待写
        smilestext_list = []

        text = mask = None
        prompt_text = prompt_mask = None

        # if self.data_type == 'para':  # paragraph-level
        text, mask = self.tokenizer_text(text_list[0])
        prompt_text, prompt_mask = self.tokenizer_prompt(prompt_text_list[0])

        # LiuC:
        data_graph_text.text = text
        data_graph_text.text_mask = mask
        data_graph_text.prompt_text = prompt_text
        data_graph_text.prompt_mask = prompt_mask

        # return data_graph, text.squeeze(0), mask.squeeze(0)  # , index
        return data_graph_text
        # return graph, text.squeeze(0), mask.squeeze(0)  # , index
        # return text.squeeze(0), mask.squeeze(0)


    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

    def tokenizer_prompt(self, prompt_text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=prompt_text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.prompt_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask



class KVSentDataset(Dataset):
    # def __init__(self, args, mode):
    def __init__(self, args, ids, scaf):
        super(KVSentDataset, self).__init__()
        self.text_max_len = args.text_max_len
        # self.mode = mode
        self.graph_name_list = os.listdir(f'data/kv_data/test/graph')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(f'data/kv_data/test/text')
        self.text_name_list.sort()
        # LiuC: 'all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/'
        self.tokenizer = BertTokenizer.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/')
        # self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        # self.data_type = args.data_type
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0
        # self.cor.append(cnt)
        for graph_name, text_name in zip(self.graph_name_list, self.text_name_list):

            text_path = os.path.join(f'data/kv_data/test/text', text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 500:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt += 1
        self.cor.append(cnt)
        np.save(f'{args.output_path}/cor.npy', self.cor)

    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device(f'cuda:1')


    # graph_encoder = GraphormerModel.from_pretrained('../all_checkpoints/graphormer_pretrained/graphormer-base-pcqm4mv1')
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

    print('##############')
    prompt_encoder = PromptEncoder(pretrained=True)
    prompt_ckpt = torch.load('../all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
    if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in prompt_ckpt:
        pretrained_dict = {"main_model." + k[20:]: v for k, v in prompt_ckpt.items()}
    elif 'bert.embeddings.word_embeddings.weight' in prompt_ckpt:
        pretrained_dict = {"main_model." + k[5:]: v for k, v in prompt_ckpt.items()}
    else:
        pretrained_dict = {"main_model." + k[12:]: v for k, v in prompt_ckpt.items()}

    prompt_encoder.load_state_dict(pretrained_dict, strict=False)
    for name, param in prompt_encoder.named_parameters():
        print(name)

    # LiuC: 通过text文件名中的序号 获取ids
    ids = []
    # text_name_list = os.listdir("../data/kv_data/text_15k")
    text_name_list = os.listdir("../preprocessing/PubChemSTM/PubChemSTM_process/text_PubChemSTM")
    for text_name in text_name_list:
        text_id = re.split('[_.]', text_name)[1]
        text_id = int(text_id)
        ids.append(text_id)
    ids.sort()
    #print(ids) [1, 2, 3, 4, 5, 6, 7, ..., 14999, 15000]
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

    # train_dataloader = torch_geometric.loader.DataLoader(TrainSet, sampler=train_sampler,
    #                                                    batch_size=args.batch_size,
    #                                                    num_workers=4, pin_memory=True, drop_last=True
    #                                                      )
    # dev_dataloader = torch_geometric.loader.DataLoader(DevSet, shuffle=False,
    #                                                  batch_size=args.batch_size,
    #                                                  num_workers=4, pin_memory=True, drop_last=True
    #                                                    )
    # test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
    #                                                   batch_size=args.batch_size,
    #                                                   num_workers=4, pin_memory=True, drop_last=True
    #                                                     )  # True



    # LiuC: use from torch.utils.data import Dataset, DataLoader
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                                         batch_size=args.batch_size,
                                                         num_workers=4, pin_memory=True, drop_last=True,
                                                         collate_fn = partial(collator, max_node=128,
                                                                              multi_hop_max_dist=20, spatial_pos_max=20))
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                                       batch_size=args.batch_size,
                                                       num_workers=4, pin_memory=True, drop_last=True,
                                                       collate_fn=partial(collator, max_node=128,
                                                                          multi_hop_max_dist=20, spatial_pos_max=20))
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                                        batch_size=args.batch_size,
                                                        num_workers=4, pin_memory=True, drop_last=True,
                                                        collate_fn=partial(collator, max_node=128,
                                                                           multi_hop_max_dist=20, spatial_pos_max=20))



    for epoch in range(args.epoch):
        print(args.epoch)
        for idx, batch in enumerate((train_dataloader)):
            # print(f"Batch {idx}: {batch}")

            # graph, text, mask = batch
            # print("graph:",graph)
            # print("text:", text)
            # print("text_mask", mask)

            # graph
            # idx = batch.idx
            attn_bias = batch.attn_bias
            attn_edge_type = batch.attn_edge_type
            spatial_pos = batch.spatial_pos
            in_degree = batch.in_degree
            out_degree = batch.out_degree
            x = batch.x
            edge_input = batch.edge_input
            # text
            text = batch.text
            text_mask = batch.text_mask
            text_batch = torch.cat(text, dim=0)
            text_mask_batch = torch.cat(text_mask, dim=0)
            # prompt
            prompt_text = batch.prompt_text
            prompt_mask = batch.prompt_mask
            prompt_text_batch = torch.cat(prompt_text, dim=0)
            prompt_mask_batch = torch.cat(prompt_mask, dim=0)

            # batch_idx,attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input = batch

            # LiuC: process  text tuple

            # LiuC: solve: data = data.to(device)
            attn_bias = attn_bias.to(device)
            attn_edge_type = attn_edge_type.to(device)
            spatial_pos = spatial_pos.to(device)
            in_degree = in_degree.to(device)
            out_degree = out_degree.to(device)
            x = x.to(device)
            edge_input = edge_input.to(device)

            text_batch = text_batch.to(device)
            text_mask_batch = text_mask_batch.to(device)

            prompt_text_batch = prompt_text_batch.to(device)
            prompt_mask_batch = prompt_mask_batch.to(device)

            prompt_emb, prompt_extended_mask = prompt_encoder(prompt_text_batch, prompt_mask_batch)

            # LiuC:
            # print("text encoder begin")------

            text_feature = text_encoder(text_batch, text_mask_batch, prompt_emb, prompt_extended_mask)
            print(f"text_feature_{idx}", text_feature)

            # text_feature = model.text_proj_head(text_feature)

            # print("graph encoder begin")
            graph_feature = graph_encoder(x, attn_bias, attn_edge_type, spatial_pos,
                                          in_degree, out_degree, edge_input, prompt_emb)
            print(f"graph_feature_{idx}", graph_feature)
            # graph_feature = model.graph_proj_head(graph_feature)

















