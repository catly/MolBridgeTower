import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader as torch_DataLoader
# from torch_geometric.loader import DataLoader as pyg_DataLoader

from MolTox.MoleculeNet_Graph import MoleculeNetGraphDataset, CustomDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from functools import partial
from MolTox.collator_prop import collator, Batch

from model.MolBT_gt import BT_Config

import torch.multiprocessing

from MolTox.splitters import scaffold_split
from MolTox.utils import get_num_task_and_type, get_molecule_repr_MoleculeSTM
# from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
# from Molprop_dataset.molecule_gnn_model import GNN, GNN_graphpred
from MolTox.molBT_graph_model_1022 import Graph_pred

# 设置共享策略
torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def eval_classification(model, device, loader):
    model.eval()
    linear_model.eval()
    y_true, y_scores = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):

        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()
        y = batch.y
        y = torch.stack(y)
        y = y.view(pred.shape).to(device).float()

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print("{} is invalid".format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores





@torch.no_grad()
def eval_regression(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):

        pred = model(batch)
        pred = pred.float()

        y = batch.y.view(pred.shape).to(device).float()

        y_true.append(y)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--training_mode", type=str, default="fine_tuning", choices=["fine_tuning", "linear_probing"])
    parser.add_argument("--molecule_type", type=str, default="Graph", choices=["SMILES", "Graph"])

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="data")
    # parser.add_argument("--dataset", type=str, default="bace")
    parser.add_argument("--dataset", type=str, default="Ames")
    parser.add_argument("--split", type=str, default="scaffold")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)  # 100
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--schedule", type=str, default="cycle")
    parser.add_argument("--warm_up_steps", type=int, default=10)


    ########## for Graphormer ##########
    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    parser.add_argument('--graph_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--drop_ratio', default=0.1, type=float)
    parser.add_argument('--projection_dim', type=int, default=256)

    ########## for saver ##########
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    # parser.add_argument("--input_model_path", type=str, default='Mol_Toxpred/pretrained/MoMu/MoMu-S.ckpt')
    parser.add_argument("--input_model_path", type=str, default=None)
    # parser.add_argument("--output_model_dir", type=str, default='save_model/task4_tox/v1')

    args = parser.parse_args()
    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device))
        # if torch.cuda.is_available() else torch.device("cpu")

    num_tasks, task_mode = get_num_task_and_type(args.dataset)
    dataset_folder = os.path.join(args.dataspace_path, "TDCommons", args.dataset)


    data_processed_path = os.path.join(args.dataspace_path, "TDCommons", args.dataset, "processed",
                                       "data_processed.pt")
    dataset = CustomDataset(data_processed_path)
    use_pyg_dataset = False

    assert args.split == "scaffold"
    print("split via scaffold")
    smiles_list = pd.read_csv(
        dataset_folder + "/processed/smiles.csv", header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=0.8,
        frac_valid=0.1, frac_test=0.1, pyg_dataset=use_pyg_dataset)

    # train_loader = dataloader_class(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_loader = dataloader_class(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = dataloader_class(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=0, pin_memory=False, drop_last=False,
                              collate_fn=partial(collator, max_node=512,
                                                 multi_hop_max_dist=20, spatial_pos_max=20))
    val_loader = DataLoader(valid_dataset, shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=0, pin_memory=False, drop_last=False,
                            collate_fn=partial(collator, max_node=512,
                                               multi_hop_max_dist=20, spatial_pos_max=20))
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=0, pin_memory=False, drop_last=False,
                             collate_fn=partial(collator, max_node=512,
                                                multi_hop_max_dist=20, spatial_pos_max=20))

    test_ckpt_path = 'save_model/task4_tox/Ames/1022_Miss_1M_200k/Ames_model_best.pth'

    test_ckpt = torch.load(test_ckpt_path)

    bt_config = BT_Config()
    model = Graph_pred(
        config=bt_config,
        temperature=args.temperature,
        drop_ratio=args.drop_ratio,
        graph_hidden_dim=args.graph_hidden_dim,
        bert_hidden_dim=args.bert_hidden_dim,
        # bert_pretrain,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_tasks=num_tasks
    )
    # model.load_state_dict(model_ckpt, False)
    model.load_state_dict(test_ckpt['model'])
    molecule_dim = args.graph_hidden_dim
    model.to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    linear_model = nn.Linear(molecule_dim, num_tasks).to(device)

    # train_func = train_classification
    eval_func = eval_classification
    train_roc = train_acc = val_roc = val_acc = 0
    test_roc, test_acc, test_target, test_pred = eval_func(model, device, test_loader)
    print("train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc, val_roc, test_roc))

