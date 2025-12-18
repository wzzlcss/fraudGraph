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

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='amazon')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gat", help="GNN layer, (default: gcn)")
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
