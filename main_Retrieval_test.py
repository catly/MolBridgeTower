import argparse
from config_RT import parse_args

import random
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import sys

# LiuC:
from data_provider.GraST_dataset_gts import GraSTMatchDataset, KVSentDataset
# from data_provider.sent_dataset import KVSentDataset
from torch.utils.data import Dataset, DataLoader
from data_provider.collator_gts import collator, Batch
import torch_geometric

from model.contrastive_BTT_gt import MolBT_CL
# from model.contrastive_BTT_gt_Miss_1M_200k import MolBT_CL

from optimization import BertAdam, warmup_linear
from torch.utils.data import RandomSampler
import os
import re
import statistics

# LiuC:
from functools import partial
# LiuC:
import logging
import time



def prepare_model_and_optimizer(args, device):

    ckpt_path = 'save_model/pretrain_1020_Miss_1M_200k/model_20_42.ckpt'

    model_ckpt = torch.load(ckpt_path)
    # print(model_ckpt)

    model = MolBT_CL(
        temperature=args.temperature,
        drop_ratio=args.drop_ratio,
        graph_hidden_dim=args.graph_hidden_dim,
        bert_hidden_dim=args.bert_hidden_dim,
        # bert_pretrain,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # model = GraSTSimclr()
    model.load_state_dict(model_ckpt)

    # if args.mode == 'linear':
    #     for p in model.graph_encoder.parameters():
    #         p.requires_grad = False
    #     for p in model.text_encoder.parameters():
    #         p.requires_grad = False

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        weight_decay=args.weight_decay,
        lr=args.lr,
        warmup=args.warmup,
        t_total=args.total_steps,
    )

    return model, optimizer


# LiuC: dev_dataloader
def Eval(model, dataloader, device, args):
    model.eval()
    with torch.no_grad():
        acc1 = 0
        acc2 = 0
        allcnt = 0
        graph_rep_total = None
        text_rep_total = None
        for idx, batch in enumerate((dataloader)):

            text_rep, graph_rep, loss = model.training_step(batch_idx=idx, batch=batch)

            scores1 = torch.cosine_similarity(
                graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]),
                text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            scores2 = torch.cosine_similarity(
                text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]),
                graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            argm1 = torch.argmax(scores1, axis=1)
            argm2 = torch.argmax(scores2, axis=1)

            acc1 += sum((argm1 == torch.arange(argm1.shape[0]).to(device)).int()).item()
            acc2 += sum((argm2 == torch.arange(argm2.shape[0]).to(device)).int()).item()

            allcnt += argm1.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), axis=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save(f'{args.output_path}/graph_rep.npy', graph_rep_total.cpu())
    np.save(f'{args.output_path}/text_rep.npy', text_rep_total.cpu())

    return acc1 / allcnt, acc2 / allcnt


# get every sentence's rep
def CalSent(model, dataloader, device, args):
    model.eval()
    with torch.no_grad():
        text_rep_total = None
        for batch in (dataloader):
            text, mask = batch
            text = text.to(device)
            mask = mask.to(device)
            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)

            if text_rep_total is None:
                text_rep_total = text_rep
            else:
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save(f'{args.output_path}/text_rep.npy', text_rep_total.cpu())




def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def main():

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.device}')
    model, optimizer = prepare_model_and_optimizer(args, device)

    ids = []
    text_name_list = os.listdir("data/kv_data/text_15k")
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

    TestSet = GraSTMatchDataset(args, ids, seq)
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=0, pin_memory=False, drop_last=True,
                                 collate_fn=partial(collator, max_node=128,
                                                    multi_hop_max_dist=20, spatial_pos_max=20))

    global_step = 0
    tag = True
    best_acc = 0
    start_time = time.time()


    acc1, acc2 = Eval(model, test_dataloader, device, args)
    print('Test Acc1:', round(acc1, 4))
    print('Test Acc2:', round(acc2, 4))
    graph_rep = torch.from_numpy(np.load(f'{args.output_path}/graph_rep.npy'))
    text_rep = torch.from_numpy(np.load(f'{args.output_path}/text_rep.npy'))
    graph_len = graph_rep.shape[0]
    text_len = text_rep.shape[0]

    # LiuC: (G-to-T Retrieval)
    score1 = torch.zeros(graph_len, graph_len)
    for i in range(graph_len):
        score1[i] = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
    rec1 = []
    for i in range(graph_len):
        a, idx = torch.sort(score1[:, i])
        for j in range(graph_len):
            if idx[-1 - j] == i:
                rec1.append(j)
                break
    rec_1 = sum((np.array(rec1) < 20).astype(int)) / graph_len
    print(f'Rec@20 1: {round(rec_1, 4)}')
    rec1 = sum((np.array(rec1) < 20).astype(int)) / graph_len

    # LiuC: (T-to-G Retrieval)
    score2 = torch.zeros(graph_len, graph_len)
    for i in range(graph_len):
        score2[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
    rec2 = []
    for i in range(graph_len):
        a, idx = torch.sort(score2[:, i])
        for j in range(graph_len):
            if idx[-1 - j] == i:
                rec2.append(j)
                break
    rec_2 = sum((np.array(rec2) < 20).astype(int)) / graph_len
    print(f'Rec@20 2: {round(rec_2, 4)}')
    rec2 = sum((np.array(rec2) < 20).astype(int)) / graph_len
    return acc1, acc2, rec1, rec2



if __name__ == "__main__":
    args = parse_args()
    print(args)
    setup_logging()
    # if not os.path.exists(args.output_path):
    #     os.mkdir(args.output_path)
    acc1_values = []
    acc2_values = []
    rec1_values = []
    rec2_values = []

    # !zero-shot (finetune)
    for seed in [42]:
        args.seed = seed
        print(f'seed:{args.seed}')
        acc1, acc2, rec1, rec2 = main()
        acc1_values.append(acc1)
        acc2_values.append(acc2)
        rec1_values.append(rec1)
        rec2_values.append(rec2)

    acc1_mean = statistics.mean(acc1_values)
    acc1_stddev = statistics.stdev(acc1_values)
    acc2_mean = statistics.mean(acc2_values)
    acc2_stddev = statistics.stdev(acc2_values)
    rec1_mean = statistics.mean(rec1_values)
    rec1_stddev = statistics.stdev(rec1_values)
    rec2_mean = statistics.mean(rec2_values)
    rec2_stddev = statistics.stdev(rec2_values)

    # import pdb;pdb.set_trace()
    logging.basicConfig(filename=f'./logs/mode_{args.mode}_data_type_{args.data_type}_logs.txt', level=logging.INFO,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('*' * 50)
    logging.info(args.init_checkpoint)
    # logging.info(f'values:{values}')

    logging.info(f'acc1_mean: {acc1_mean:.4f}')
    logging.info(f'acc1_stddev: {acc1_stddev:.4f}')
    logging.info(f'acc2_mean: {acc2_mean:.4f}')
    logging.info(f'acc2_stddev: {acc2_stddev:.4f}')
    logging.info(f'rec1_mean: {rec1_mean:.4f}')
    logging.info(f'rec1_stddev: {rec1_stddev:.4f}')
    logging.info(f'rec2_mean: {rec2_mean:.4f}')
    logging.info(f'rec2_stddev: {rec2_stddev:.4f}')
    logging.info('*' * 50)







