from tqdm import tqdm
import numpy as np

import torch
import argparse
from torch import Tensor

from graph_utils.load_seq_event import *
from gae.model_MaskGAE import *
from graph_utils.edge_selection import *

from torch.utils.data import DataLoader
from utils import compute_batch_auc, compute_batch_pr_auc

import logging
import datetime
import os

import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score


# from torch_geometric.utils import add_self_loops

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='amazon')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=64, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--heads', type=int, default=2, help='Number of gat heads. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.1, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.1, help='Dropout probability of decoder. (default: 0.3)')
# parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
# parser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
# parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge')
# parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
# parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
# parser.add_argument("--save_path", nargs="?", default="MaskGAE-LinkPred.pt", help="save path for model. (default: MaskGAE-LinkPred.pt)")
# parser.add_argument("--device", type=int, default=0)
# training option
# parser.add_argument("--use_special_negative", action='store_true')
# parser.add_argument("--sample_negative", action='store_true')
# parser.add_argument("--sample_ratio", type=int, default=5, help='1 positive vs this many negative')
# parser.add_argument("--exp", type=str, default="exp1")
# parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

### logging
run_id = np.random.randint(10000, 99999)
now = datetime.datetime.now()
output_path = os.getcwd()
output_path = os.path.join(
    output_path, "runs", "run_" + str(now.day) + "." + str(now.month) +
        "." + str(now.year) + "_" + str(run_id))
os.makedirs(os.path.join(output_path, "models"))
logging.basicConfig(
    filename=os.path.join(output_path, "log_" + str(run_id) + ".txt"), filemode='w',
    level=logging.INFO, format='[%(levelname)s]%(message)s')
for arg in sorted(vars(args)):
    logging.info("{0}: {1}".format(arg, getattr(args, arg)))

logging.info("----------")

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

# all_negative_edges_train = find_all_not_existing_edges(train_iter, vocab_dict)
# all_negative_edges_valid = find_all_not_existing_edges(valid_iter, vocab_dict)

train_dataloader, valid_dataloader, test_dataloader, feat, N = prepare_seq(
    data_name=args.dataset_name)

# node input feature: trained embedding from BERT
feat = feat.to(device).to(dtype=torch.float32)

# # find common positive among training sequence
# common_positive = find_all_edges(train_eventSeq)
# common_positive = remove_self_loop(common_positive)
# common_negative = find_all_negative_edges(
#     common_positive, n, exclude_diag=True)
# common_negative = remove_self_loop(common_negative)

# total_positive = 0.0
# total_size = (n*n) * len(valid_eventSeq)
# for seq in valid_eventSeq:
#     num_positive = int(len(seq)/2) - 1
#     total_positive += num_positive

n_feat = feat.shape[1]
encoder = GNNEncoder(
    in_channels=n_feat, hidden_channels=args.encoder_channels, out_channels=args.hidden_channels,
    num_layers=args.encoder_layers, dropout=args.encoder_dropout,
    bn=args.bn, layer=args.layer, activation=args.encoder_activation,
    heads=args.heads)

edge_decoder = EdgeDecoder(
    args.hidden_channels, args.decoder_channels,
    num_layers=args.decoder_layers, dropout=args.decoder_dropout)

# mask = MaskEdge(p=args.p)

model = CondGAE(encoder, edge_decoder).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# inductive: multiple graphs
# input to the model: edge_index_unmasked; n, feat; edge_index_masked_for_loss 
# n_train_sample = len(train_dataloader)

# all_edges_n = all_edges(n)
# all_edges_n = torch.tensor(all_edges_n).to(device).to(dtype=torch.long)

# edge_index_empty = torch.empty(2, 0, dtype=torch.long)
# self_loop_only, _ = add_self_loops(edge_index_empty, num_nodes = n)
# self_loop_only = self_loop_only.to(device).to(dtype=torch.long)

def find_nontrivial_negative(input_edge, output_edge, n, common_positive):
    # new negative compared with t-1
    new_negative = find_new_negative(input_edge, output_edge, n)
    # curr negative that is in common positive
    curr_negative_in_common_positive, _ = curr_negative_vs_common_positive(
        output_edge, common_positive, n)
    # negative that is relevant to the current graph
    # relevant_negative = neg_edges_relevant(output_edge, n)
    return new_negative, curr_negative_in_common_positive #, relevant_negative

def sample_edges(full_set, target_size=500):
    if full_set.shape[1] <= target_size:
        return full_set
    idx = torch.randint(
    full_set.shape[1], (target_size,), device=full_set.device)
    return full_set[:, idx]

def compute_batch_false_positive(pred_for_negative):
    all_pred = torch.cat(pred_for_negative, dim=0).squeeze()
    FP = (all_pred == 1).sum().item()
    return FP / len(all_pred)

def compute_batch_false_negative(pred_for_positive):
    all_pred = torch.cat(pred_for_positive, dim=0).squeeze()
    FN = (all_pred == 0).sum().item()
    return FN / len(all_pred)

def evaluate_negative(z, negative):
    negative = torch.tensor(negative).to(device).to(dtype=torch.long)
    # negative = sample_edges(
    #     negative, target_size=500)
    return model.evaluate_certain(z, negative)

def evaluate_positive(z, positive):
    positive = torch.tensor(positive).to(device).to(dtype=torch.long)
    return model.evaluate_certain(z, positive)

# train_loss = []
# valid_auc = []
# valid_pr_auc = []
# fp_new_negative = []
# fp_curr_negative_in_common_positive = []

# fn_output_edge = []
# fn_new_in_output_edge = []
# # fp_pred_relevant_negative = []

def train_on_one_sequence(seq):
    input_edge, output_edge = seq
    all_negatives = find_all_negative_edges(
        output_edge, N, exclude_diag=True)
    all_negatives = keep_unique(all_negatives)
    # prepare input
    all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
    input_edge = torch.tensor(input_edge).to(device).to(dtype=torch.long)
    output_edge = torch.tensor(output_edge).to(device).to(dtype=torch.long)
    loss = model(feat, input_edge, output_edge, all_negatives)
    return loss

def eval_static(seq):
    input_edge, output_edge = seq
    all_negatives = find_all_negative_edges(
        output_edge, N, exclude_diag=True)
    all_negatives = keep_unique(all_negatives)
    # prepare input
    all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
    input_edge = torch.tensor(input_edge).to(device).to(dtype=torch.long)
    output_edge = torch.tensor(output_edge).to(device).to(dtype=torch.long)
    y_cond, pred_cond = model.evaluate(feat, input_edge, output_edge, all_negatives)
    return pred_cond.detach().cpu(), y_cond.cpu()

def eval_dataloader(dataloader):
    y_list_cond = []; pred_list_cond = []
    for _, seq in enumerate(dataloader, start=1):
        print(_, end='\r')
        predicts, labels = eval_static(seq)
        y_list_cond.append(labels)
        pred_list_cond.append(predicts)
    ####
    auc_cond = compute_batch_auc(y_list_cond, pred_list_cond)
    pr_auc_cond = compute_batch_pr_auc(y_list_cond, pred_list_cond)
    return auc_cond, pr_auc_cond

n_train_sample = len(train_dataloader)
assert n_train_sample % args.batch_size == 0, \
    f"train_dataloader length ({n_train_sample}) must be divisible by batch_size ({args.batch_size})"

train_loss = []
valid_auc_list = []; valid_pa = []
test_auc_list = []; test_pa = []
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    # training
    for step, train_seq in enumerate(train_dataloader, start=1):
        print(step, end='\r')
        loss = train_on_one_sequence(train_seq)
                # define training norm
        loss = loss / args.batch_size
        loss.backward()
        running_loss += loss.item()
        # train_losses.append(loss.item())
        if step % args.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            running_loss = running_loss / args.batch_size # avg batch loss
            train_loss.append(running_loss)
            logging.info(f"Avg Batch Loss ({step} / {n_train_sample}): {running_loss}")
            running_loss = 0.0
    # validation after each epoch
    model.eval()
    valid_auc_cond, valid_pr_auc_cond = eval_dataloader(valid_dataloader)
    test_auc_cond, test_pr_auc_cond = eval_dataloader(test_dataloader)
    logging.info(f"Step: {epoch+1}; validation roc_auc (true condition): {valid_auc_cond}; pr auc: {valid_pr_auc_cond}")
    logging.info(f"Step: {epoch+1}; test roc_auc (true condition): {test_auc_cond}; pr auc: {test_pr_auc_cond}")
    valid_auc_list.append(valid_auc_cond); valid_pa.append(valid_pr_auc_cond)
    test_auc_list.append(test_auc_cond); test_pa.append(test_pr_auc_cond)
