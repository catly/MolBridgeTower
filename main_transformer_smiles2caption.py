from lib2to3.pgen2 import token
import torch_geometric
import torch
from transformers import AutoTokenizer, LogitsProcessorList, BeamSearchScorer, BertTokenizer, T5Tokenizer
from torch import nn
import torch.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
# from torch.utils.data.dataloader import default_collate
from transformers.optimization import get_linear_schedule_with_warmup

import numpy as np
import pickle

import argparse
import sys

# from Molcaption.model.GinT5 import GinDecoder
from Molcaption.model.MolBTT5 import MolBTDecoder
# from Molcaption.model.GraphormerT5 import MolBTDecoder

# from model.gin_model import GNN
from Molcaption.dataloader import TextMoleculeReplaceDataset
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from functools import partial
from Molcaption.collator_caption import collator, Batch

#import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', help='mode')
# parser.add_argument('--mode', type=str, default='train', help='mode')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size')
# parser.add_argument('--nlayers', type=int, default=6, help='number of layers')
# parser.add_argument('--emb_size', type=int, default=512, help='input dimension size')
parser.add_argument('--max_length', type=int, default=512, help='max length')
parser.add_argument('--max_smiles_length', type=int, default=512, help='max smiles length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--nhead', type=int, default=8, help='num attention heads')

# parser.add_argument('--MoMuK', default=False, action='store_true')
parser.add_argument('--model_size', type=str, default='small')
parser.add_argument('--data_path', type=str, default='Molcaption/data/', help='path where data is located =')
parser.add_argument('--saved_path', type=str, default='save_model/task3_caption/train/1022_Miss_1M_200k/', help='path where weights are saved')

parser.add_argument('--output_file', type=str, default='save_model/task3_caption/output_small/1022_Miss_1M_200k/small_out_v1_3e_5_1022.txt', help='path where test generations are saved')


args = parser.parse_args()

runseed = 100
torch.manual_seed(runseed)
np.random.seed(runseed)
device = torch.device(f"cuda:1")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(runseed)

tokenizer = T5Tokenizer.from_pretrained("Molcaption/MolT5_pretrained/molt5-small-smiles2caption/", model_max_length=512)

train_data = TextMoleculeReplaceDataset(args.data_path, 'train', tokenizer)
val_data = TextMoleculeReplaceDataset(args.data_path, 'validation', tokenizer)
test_data = TextMoleculeReplaceDataset(args.data_path, 'test', tokenizer)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler,
                          batch_size=args.batch_size,
                          num_workers=4, pin_memory=False, drop_last=False,
                          collate_fn=partial(collator, max_node=512,
                                             multi_hop_max_dist=20, spatial_pos_max=20))
val_dataloader = DataLoader(val_data, shuffle=True,
                        batch_size=args.batch_size,
                        num_workers=4, pin_memory=False, drop_last=False,
                        collate_fn=partial(collator, max_node=512,
                                           multi_hop_max_dist=20, spatial_pos_max=20))
test_dataloader = DataLoader(test_data, shuffle=False,
                         batch_size=args.batch_size,
                         num_workers=4, pin_memory=False, drop_last=False,
                         collate_fn=partial(collator, max_node=512,
                                            multi_hop_max_dist=20, spatial_pos_max=20))


model = MolBTDecoder(has_graph=True, model_size=args.model_size).to(device)

if args.mode == 'test':
    state_dict = torch.load('save_model/task3_caption/train/1022_Miss_1M_200k/MolBTT5_smiles2caption_small_v1_3e_5_1022.pt')
    model.load_state_dict(state_dict)

if args.mode == 'train':
    for p in model.named_parameters():
    	if p[1].requires_grad:
            print(p[0])

    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(pg, lr=args.lr)

MAX_LENGTH = args.max_length


def train_epoch(dataloader, model, optimizer, epoch):
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    
    # model.train()
    losses = 0
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # for j, d in tqdm(enumerate(dataloader), total=len(dataloader)):
    for j, batch in tqdm(enumerate(dataloader)):
        # model.zero_grad()

        smiles_tokens_ = tokenizer(batch.smiles, padding=True, truncation=True, return_tensors="pt")
        smiles_tokens = smiles_tokens_['input_ids'].to(device)
        src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
        
        # text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
        text_tokens_ = tokenizer(batch.description, padding=True, truncation=True, return_tensors="pt")
        text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
        label = text_tokens_['input_ids'].to(device)  # caption

        label = label.masked_fill(~text_mask.bool(), -100)


        loss = model(batch, smiles_tokens, src_padding_mask, text_mask, label)

        if j % 300 == 0:
            print('total steps: {}, step: {}, loss: {}'.format(epoch*len(dataloader) + j, j, loss))
        # print('total steps: {}, step: {}, loss: {}'.format(epoch * len(dataloader) + j, j, loss))

        loss.backward()
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()

        losses += loss.item()

    return losses / len(dataloader)


def eval(dataloader, model, epoch):
    model.eval()
    losses = 0
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for j, batch in tqdm(enumerate(dataloader)):


            smiles_tokens_ = tokenizer(batch.smiles, padding=True, truncation=True, return_tensors="pt")
            smiles_tokens = smiles_tokens_['input_ids'].to(device)
            src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
            
            text_tokens_ = tokenizer(batch.description, padding=True, truncation=True, return_tensors="pt")
            text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
            label = text_tokens_['input_ids'].to(device)  # caption

            label = label.masked_fill(~text_mask.bool(), -100)

            loss = model(batch, smiles_tokens, src_padding_mask, text_mask, label)
            losses += loss.item()
            if j % 100 == 0:
                print('val total steps: {}, step: {}, val loss: {}'.format(epoch*len(dataloader) + j, j, loss))

    
    return losses/len(dataloader)


if args.mode == 'train':
    # my_model.train()
    min_val_loss = 10000
    for i in range(args.epochs):
        print('Epoch:', i)
        train_epoch(train_dataloader, model=model, optimizer=optimizer, epoch=i)
        print("Begin Eval##########")
        val_loss = eval(val_dataloader, model=model, epoch=i)
        print(min_val_loss)
        print(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("--------------------save model--------------------")
            torch.save(model.state_dict(), args.saved_path + 'MolBTT5_smiles2caption_small_v1_3e_5_1022.pt')




if args.mode == 'test':
    model.eval()
    smiles = []
    test_outputs = []
    test_gt = []
    with torch.no_grad():
        # for j, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        for j, batch in tqdm(enumerate(test_dataloader)):
            # real_text = d['description']
            # graph = d['graph'].to(device)
            real_text = batch.description

            # smiles_tokens_ = tokenizer(d['smiles'], padding=True, truncation=True, return_tensors="pt")
            smiles_tokens_ = tokenizer(batch.smiles, padding=True, truncation=True, return_tensors="pt")
            smiles_tokens = smiles_tokens_['input_ids'].to(device)
            src_padding_mask = smiles_tokens_['attention_mask'].to(device)  # encoder input mask
            
            # text_tokens_ = tokenizer(d['description'], padding=True, truncation=True, return_tensors="pt")
            # text_mask = text_tokens_['attention_mask'].to(device)  # caption mask, decoder input mask
            # label = text_tokens_['input_ids'].to(device)  # caption
          
            outputs = model.translate(batch, smiles_tokens, src_padding_mask, tokenizer)

            # print(outputs)
            # break

            # smiles.extend(d['smiles'])
            print(len(batch.smiles))

            smiles.extend(batch.smiles)
            test_gt.extend(real_text)
            test_outputs.extend(outputs)


    with open(args.output_file, 'w') as f:
        f.write('SMILES' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
        for smi, rt, ot in zip(smiles, test_gt, test_outputs):
            f.write(smi + '\t' + rt + '\t' + ot + '\n')

    print(len(smiles))
    print(len(test_gt))
    print(len(test_outputs))

