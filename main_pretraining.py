import argparse
from config import parse_args

import random
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import sys

# LiuC:
# from data_provider.GraST_dataset import GraSTMatchDataset, KVSentDataset, GraSTPretrain_STM
from data_provider.GraST_dataset_gts import GraSTMatchDataset, KVSentDataset, GraSTPretrain_STM
# from data_provider.sent_dataset import KVSentDataset
from torch.utils.data import Dataset, DataLoader
from data_provider.collator_gts import collator, Batch
import torch_geometric
# from model.contrastive_GraST_0313 import GraSTSimclr
from model.contrastive_BTT import MolBT_CL
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
            # aug, text, mask = batch
            # aug = aug.to(device)
            #
            # text = text.to(device)
            # mask = mask.to(device)
            # graph_rep = model.graph_encoder(aug)
            # graph_rep = model.graph_proj_head(graph_rep)
            #
            # text_rep = model.text_encoder(text, mask)
            # text_rep = model.text_proj_head(text_rep)
            text_rep, graph_rep, loss = model.training_step(batch_idx=idx, batch=batch)
            # text_rep, graph_rep, _ = model.training_step(batch_idx=idx, batch=batch)

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


# def Contra_Loss(logits_des, logits_smi, margin, device):
#     scores = torch.cosine_similarity(
#         logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]),
#         logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
#     diagonal = scores.diag().view(logits_smi.size(0), 1)
#     d1 = diagonal.expand_as(scores)
#     d2 = diagonal.t().expand_as(scores)
#
#     cost_des = (margin + scores - d1).clamp(min=0)
#     cost_smi = (margin + scores - d2).clamp(min=0)
#
#     # clear diagonals
#     mask = torch.eye(scores.size(0)) > .5
#     I = Variable(mask)
#     if torch.cuda.is_available():
#         I = I.to(device)
#     cost_des = cost_des.masked_fill_(I, 0)
#     cost_smi = cost_smi.masked_fill_(I, 0)
#
#     # keep the maximum violating negative for each query
#     # if self.max_violation:
#     cost_des = cost_des.max(1)[0]
#     cost_smi = cost_smi.max(0)[0]
#
#     return cost_des.sum() + cost_smi.sum()


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

    # LiuC: 通过text文件名中的序号 获取ids
    ids = []
    # text_name_list = os.listdir("data/kv_data/text_15k")
    text_name_list = os.listdir("preprocessing/PubChemSTM/PubChemSTM_process/text_PubChemSTM")
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
    # scaf.append(seq[:7 * k])
    # LiuC: 仅使用200k数据集的一半
    # scaf.append(seq[:5 * k])
    # scaf.append(seq[7 * k:8 * k])
    # scaf.append(seq[8 * k:])

    scaf.append(seq[:9 * k])
    scaf.append(seq[9 * k:])

    TrainSet = GraSTPretrain_STM(args, ids, scaf[0])
    DevSet = GraSTPretrain_STM(args, ids, scaf[1])
    # TestSet = GraSTPretrain_STM(args, ids, scaf[2])
    train_sampler = RandomSampler(TrainSet)

    # LiuC: use from torch.utils.data import Dataset, DataLoader
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  collate_fn=partial(collator, max_node=128,
                                                     multi_hop_max_dist=20, spatial_pos_max=20))
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=4, pin_memory=True, drop_last=True,
                                collate_fn=partial(collator, max_node=128,
                                                   multi_hop_max_dist=20, spatial_pos_max=20))
    # test_dataloader = DataLoader(TestSet, shuffle=False,
    #                              batch_size=args.batch_size,
    #                              num_workers=4, pin_memory=True, drop_last=True,
    #                              collate_fn=partial(collator, max_node=128,
    #                                                 multi_hop_max_dist=20, spatial_pos_max=20))

    global_step = 0
    tag = True
    best_acc = 0
    # start_time = time.time()
    num_total_steps = len(train_dataloader) * args.epoch

    print(args.epoch)
    # !zero-shot (finetune)
    for epoch in range(args.epoch):
        if tag == False:
            break
        print('LiuC: Epoch ', epoch)
        acc1, acc2 = Eval(model, dev_dataloader, device, args)
        print('Epoch:', epoch, ', DevAcc1:', acc1)
        print('Epoch:', epoch, ', DevAcc2:', acc2)
        if acc1 > best_acc:
            best_acc = acc1
            # torch.save(model.state_dict(), f'{args.output_path}/model.ckpt')
            torch.save(model.state_dict(), f'{args.output_path}/model_{epoch}_{seed}_{acc1}_{acc2}.ckpt')
            print('Save checkpoint ', global_step)
        acc = 0
        allcnt = 0
        sumloss = 0
        model.train()
        for idx, batch in enumerate((train_dataloader)):
            # text_rep, graph_rep, _ = model.training_step(batch_idx=idx, batch=batch)
            # loss = Contra_Loss(graph_rep, text_rep, args.margin, device)
            text_rep, graph_rep, loss = model.training_step(batch_idx=idx, batch=batch)
            print('Epoch:', epoch, ', global_step:', global_step, ', loss:', loss)
            scores = text_rep.mm(graph_rep.t())
            argm = torch.argmax(scores, axis=1)
            acc += sum((argm == torch.arange(argm.shape[0]).to(device)).int()).item()
            allcnt += argm.shape[0]
            sumloss += loss.item()
            loss.backward()
            # if idx%4==1:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step > args.total_steps:
            # if global_step > num_total_steps:
                # time_per_step = (time.time() - start_time) / max(1, global_step)
                # remaining_time = time_per_step * (num_total_steps - global_step)
                # remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                # logging.info(
                #     f"Epoch {epoch} step {global_step} eta {remaining_time}: loss {loss:.3f}")
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch:', epoch, ', Acc:', acc / allcnt, ', Loss:', sumloss / allcnt)

    # acc1, acc2 = Eval(model, dev_dataloader, device, args)
    # print('Epoch:', args.epoch, ', DevAcc1:', acc1)
    # print('Epoch:', args.epoch, ', DevAcc2:', acc2)
    # logging.info(f"Epoch {epoch} step {global_step}: loss {loss:.3f}")
    # if acc1 > best_acc:
    #     best_acc = acc1
    #     # torch.save(model.state_dict(), f'{args.output_path}/model.ckpt')
    #     torch.save(model.state_dict(), f'{args.output_path}/model_{epoch}_{seed}_{acc1}_{acc2}.ckpt')
    #     print('Save checkpoint ', global_step)
    #
    # model.load_state_dict(torch.load(f'{args.output_path}/model_{epoch}_{seed}_{acc1}_{acc2}.ckpt'))



    # DownStream Tasks: sent-level
    # acc1, acc2 = Eval(model, test_dataloader, device, args)
    # print(f"seed: {args.seed}")
    # print('Test Acc1:', acc1)
    # print('Test Acc2:', acc2)
    # graph_rep = torch.from_numpy(np.load(f'{args.output_path}/graph_rep.npy'))
    # SentSet = KVSentDataset(args, ids, scaf[2])
    # # LiuC:
    # print("KVTest OK")
    # sent_dataloader = DataLoader(SentSet, shuffle=False,
    #                              batch_size=args.batch_size,
    #                              num_workers=4, pin_memory=True, drop_last=False)  # True
    # CalSent(model, sent_dataloader, device, args)
    # graph_rep = torch.from_numpy(np.load(f'{args.output_path}/graph_rep.npy'))
    # text_rep = torch.from_numpy(np.load(f'{args.output_path}/text_rep.npy'))
    # cor = np.load(f'{args.output_path}/cor.npy')
    #
    # graph_len = graph_rep.shape[0]
    # text_len = text_rep.shape[0]
    #
    # score1 = torch.zeros(graph_len, graph_len)
    # score2 = torch.zeros(graph_len, graph_len)
    #
    # for i in range(graph_len):
    #     score = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
    #     for j in range(graph_len):
    #         total = 0
    #         for k in range(cor[j], cor[j + 1]):
    #             total += (score[k] / (cor[j + 1] - cor[j]))
    #         score1[i, j] = total
    #         # score1[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
    # rec1 = []
    # for i in range(graph_len):
    #     a, idx = torch.sort(score1[:, i])
    #     for j in range(graph_len):
    #         if idx[-1 - j] == i:
    #             rec1.append(j)
    #             break
    # print(f'Rec@20 1: {sum((np.array(rec1) < 20).astype(int)) / graph_len}')
    # rec1 = sum((np.array(rec1) < 20).astype(int)) / graph_len
    #
    # score_tmp = torch.zeros(text_len, graph_len)
    # for i in range(text_len):
    #     score_tmp[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
    # score_tmp = torch.t(score_tmp)
    #
    # for i in range(graph_len):
    #     for j in range(graph_len):
    #         total = 0
    #         for k in range(cor[j], cor[j + 1]):
    #             total += (score_tmp[i][k] / (cor[j + 1] - cor[j]))
    #         score2[i, j] = total
    #         # score2[i,j] = sum(score_tmp[i][cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
    # score2 = torch.t(score2)
    #
    # rec2 = []
    # for i in range(graph_len):
    #     a, idx = torch.sort(score2[:, i])
    #     for j in range(graph_len):
    #         if idx[-1 - j] == i:
    #             rec2.append(j)
    #             break
    # print(f'Rec@20 2: {sum((np.array(rec2) < 20).astype(int)) / graph_len}')
    # rec2 = sum((np.array(rec2) < 20).astype(int)) / graph_len
    # return acc1, acc2, rec1, rec2


    # DownStream Tasks: para-level
    # acc1, acc2 = Eval(model, test_dataloader, device, args)
    # print('Test Acc1:', round(acc1, 4))
    # print('Test Acc2:', round(acc2, 4))
    # graph_rep = torch.from_numpy(np.load(f'{args.output_path}/graph_rep.npy'))
    # text_rep = torch.from_numpy(np.load(f'{args.output_path}/text_rep.npy'))
    # graph_len = graph_rep.shape[0]
    # text_len = text_rep.shape[0]
    #
    # # LiuC: (G-to-T Retrieval)
    # score1 = torch.zeros(graph_len, graph_len)
    # for i in range(graph_len):
    #     score1[i] = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
    # rec1 = []
    # for i in range(graph_len):
    #     a, idx = torch.sort(score1[:, i])
    #     for j in range(graph_len):
    #         if idx[-1 - j] == i:
    #             rec1.append(j)
    #             break
    # rec_1 = sum((np.array(rec1) < 20).astype(int)) / graph_len
    # print(f'Rec@20 1: {round(rec_1, 4)}')
    # rec1 = sum((np.array(rec1) < 20).astype(int)) / graph_len
    #
    # # LiuC: (T-to-G Retrieval)
    # score2 = torch.zeros(graph_len, graph_len)
    # for i in range(graph_len):
    #     score2[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
    # rec2 = []
    # for i in range(graph_len):
    #     a, idx = torch.sort(score2[:, i])
    #     for j in range(graph_len):
    #         if idx[-1 - j] == i:
    #             rec2.append(j)
    #             break
    # rec_2 = sum((np.array(rec2) < 20).astype(int)) / graph_len
    # print(f'Rec@20 2: {round(rec_2, 4)}')
    # rec2 = sum((np.array(rec2) < 20).astype(int)) / graph_len
    # return acc1, acc2, rec1, rec2



if __name__ == "__main__":
    args = parse_args()
    print(args)
    setup_logging()
    # if not os.path.exists(args.output_path):
    #     os.mkdir(args.output_path)


    # !zero-shot (finetune)
    # for seed in [42]:
    for seed in [42]:
        args.seed = seed
        print(f'seed:{args.seed}')
        main()

