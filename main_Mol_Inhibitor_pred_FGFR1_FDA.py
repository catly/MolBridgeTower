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
from MolTox.collator_prop_molBT_FDA import collator, Batch

from model.MolBT_gt import BT_Config

import torch.multiprocessing

from MolTox.splitters import scaffold_split
from MolTox.utils import get_num_task_and_type, get_molecule_repr_MoleculeSTM
# from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
# from Molprop_dataset.molecule_gnn_model import GNN, GNN_graphpred
# from MolProp.molBT_graph_model import Graph_pred
from MolTox.molBT_graph_model_1022 import Graph_pred

from scipy.stats import spearmanr, pearsonr

# 设置共享策略
torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def eval_regression(model, device, loader):
    model.eval()
    y_pred = []
    smiles_list = []

    molecule_reprs = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    for step, batch in enumerate(L):
        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()

        y_pred.append(pred)
        smiles_list.extend(batch.smiles)

        molecule_reprs.append(molecule_repr.cpu())

    if len(y_pred) == 0:
        raise ValueError("Error: No predictions collected. Check your dataloader.")

    y_pred = torch.cat(y_pred, dim=0).cpu().numpy().squeeze()  # 变成 numpy 数组

    molecule_reprs = torch.cat(molecule_reprs, dim=0).cpu().numpy()  # (N, D) 形状

    np.savez("molecule_data_FGFR1_FDA.npz", molecule_repr=molecule_reprs, y_pred=y_pred)

    return y_pred, smiles_list





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--training_mode", type=str, default="fine_tuning", choices=["fine_tuning", "linear_probing"])
    parser.add_argument("--molecule_type", type=str, default="Graph", choices=["SMILES", "Graph"])

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="data")
    # parser.add_argument("--dataset", type=str, default="bace")
    parser.add_argument("--dataset", type=str, default="FDA")
    parser.add_argument("--split", type=str, default="scaffold")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)  # 100
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

    parser.add_argument("--input_model_path", type=str, default=None)

    # parser.add_argument("--output_model_dir", type=str, default='save_model/task4_tox/LD50/0728_200k')
    parser.add_argument("--output_model_dir", type=str, default='save_model/task4_tox/FGFR1')

    args = parser.parse_args()
    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device))
    print(device)

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


    train_dataset = []
    val_dataset = []
    test_dataset = dataset
    # train_dataset, val_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0, frac_valid=0, frac_test=1.0,
    #                               pyg_dataset=use_pyg_dataset)
    print("Train set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Test set size:", len(test_dataset))


    test_loader = DataLoader(test_dataset, shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=0, pin_memory=False, drop_last=False,
                                 collate_fn=partial(collator, max_node=512,
                                                    multi_hop_max_dist=20, spatial_pos_max=20))


    test_ckpt_path = 'save_model/task4_tox/FGFR1/FGFR1_model_best.pth'

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



    # Rewrite the seed by MegaMolBART
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # model = model.to(device)
    linear_model = nn.Linear(molecule_dim, num_tasks).to(device)


    eval_func = eval_regression

    y_pred, smiles_list = eval_func(model, device, test_loader)

