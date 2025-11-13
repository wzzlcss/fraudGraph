import numpy as np

import torch
import argparse
from torch import Tensor

from graph_utils.load_seq_log import *
from gae.model_MaskGAE import *

from torch.utils.data import DataLoader

import random
from sklearn.metrics import roc_auc_score

# from torch_geometric.utils import add_self_loops

parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
# parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2025, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.3, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.3, help='Dropout probability of decoder. (default: 0.3)')
# parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
# parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')
parser.add_argument('--batch_size', type=int, default=6000, help='Number of training batch size.')

# parser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge')
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 300)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-LinkPred.pt", help="save path for model. (default: MaskGAE-LinkPred.pt)")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

# all_negative_edges_train = find_all_not_existing_edges(train_iter, vocab_dict)
# all_negative_edges_valid = find_all_not_existing_edges(valid_iter, vocab_dict)

train_dataloader, valid_dataloader, train_eventSeq, feat, n = prepare_seq(
    name="tbird")
# find common positive among training sequence
common_positive = find_all_edges(train_eventSeq)
common_positive = remove_self_loop(common_positive)
common_negative = find_all_negative_edges(
    common_positive, n, exclude_diag=True)
common_negative = remove_self_loop(common_negative)

n_feat = feat.shape[1]
encoder = GNNEncoder(
    in_channels=n_feat, hidden_channels=args.encoder_channels, out_channels=args.hidden_channels,
    num_layers=args.encoder_layers, dropout=args.encoder_dropout,
    bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder = EdgeDecoder(
    args.hidden_channels, args.decoder_channels,
    num_layers=args.decoder_layers, dropout=args.decoder_dropout)

# mask = MaskEdge(p=args.p)

model = CondGAE(encoder, edge_decoder).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def compute_batch_auc(y_list, pred_list):
    all_pred = torch.cat(pred_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    all_y, all_pred = all_y.cpu().numpy(), all_pred.cpu().numpy()
    auc = roc_auc_score(all_y, all_pred)
    return auc

# inductive: multiple graphs
# input to the model: edge_index_unmasked; n, feat; edge_index_masked_for_loss 
n_train_sample = len(train_dataloader)

# all_edges_n = all_edges(n)
# all_edges_n = torch.tensor(all_edges_n).to(device).to(dtype=torch.long)

# edge_index_empty = torch.empty(2, 0, dtype=torch.long)
# self_loop_only, _ = add_self_loops(edge_index_empty, num_nodes = n)
# self_loop_only = self_loop_only.to(device).to(dtype=torch.long)

# node input feature: trained embedding from BERT
x = feat.to(device).to(dtype=torch.float32)

accum_steps = 1000
remainder = n_train_sample % accum_steps
last_chunk_size = remainder if remainder != 0 else acc_steps
last_chunk_step = n_train_sample - remainder

train_loss = []

fp_new_negative = []
fp_curr_negative_in_common_positive = []
# fp_pred_relevant_negative = []

best_auc = 0.0
best_model = None

import time

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

def evaluate_negative(z, negative):
    negative = torch.tensor(negative).to(device).to(dtype=torch.long)
    negative = sample_edges(
        negative, target_size=500)
    return model.evaluate_certain(z, negative)

# one epoch
model.train()
optimizer.zero_grad()
running_loss = 0.0
for step, batch in enumerate(train_dataloader, start=1):
    print(step, end='\r')
    # model input (one graph: unmasked all edges; pre-trained node embeddings)
    input_edge, output_edge = batch
    new_negative, curr_negative_in_common_positive = find_nontrivial_negative(
        input_edge, output_edge, n, common_positive)
    # sample negative edges
    arrays = [
        a.reshape(2, -1) if a.ndim == 1 else a
        for a in [new_negative, curr_negative_in_common_positive]
    ]
    all_negatives = np.hstack([a for a in arrays if a.size > 0])
    all_negatives = keep_unique(all_negatives)
    # prepare input
    all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
    input_edge = torch.tensor(input_edge).to(device).to(dtype=torch.long)
    output_edge = torch.tensor(output_edge).to(device).to(dtype=torch.long)
    # define training norm
    is_acc_step = ((step + 1) % accum_steps == 0)
    is_last_chunk_step = ((step + 1) >= last_chunk_step)
    demon = accum_steps if not is_last_chunk_step else remainder
    is_last_step = ((n + 1) == n_train_sample)
    # define training step
    loss = model(x, input_edge, output_edge, all_negatives, n_neg_per_pos = 1)
    loss = loss / demon
    loss.backward()
    running_loss += loss.item()
    if is_acc_step or is_last_step: # make change here
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(running_loss)
        print(f"Total avg Loss ({step+1} / {n_train_sample}): {running_loss}")
        running_loss = 0.0
        ####### validation #######
        model.eval()
        pred_new_negative = []
        pred_curr_negative_in_common_positive = []
        # pred_relevant_negative = []
        y_list_cond = []; pred_list_cond = []
        for _, batch in enumerate(valid_dataloader, start=1):
            input_edge, output_edge = batch
            new_negative, curr_negative_in_common_positive = find_nontrivial_negative(
                input_edge, output_edge, n, common_positive)
            arrays = [
                a.reshape(2, -1) if a.ndim == 1 else a
                for a in [new_negative, curr_negative_in_common_positive]
            ]
            all_negatives = np.hstack([a for a in arrays if a.size > 0])
            all_negatives = keep_unique(all_negatives)
            # prepare input
            all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
            input_edge = torch.tensor(input_edge).to(device).to(dtype=torch.long)
            output_edge = torch.tensor(output_edge).to(device).to(dtype=torch.long)
            # overall auc, downsample negatives
            num_sample = input_edge.shape[1] * 1
            sample_idx = torch.randint(
                all_negatives.shape[1], (num_sample,), device=all_negatives.device)
            neg_samples = all_negatives[:, sample_idx]
            y_cond, pred_cond = model.evaluate(x, input_edge, output_edge, neg_samples)
            y_list_cond.append(y_cond); pred_list_cond.append(pred_cond.squeeze())
            # evaluate certain kind of negative
            z = model.encoder(x, input_edge)
            if new_negative.size > 0:
                pred_new_negative.append(evaluate_negative(z, new_negative))
            if curr_negative_in_common_positive.size > 0:
                pred_curr_negative_in_common_positive.append(
                    evaluate_negative(z, curr_negative_in_common_positive)
                )
        #### compute overall stat for validation set
        fp1 = compute_batch_false_positive(pred_new_negative)
        fp_new_negative.append(fp1)
        fp2 = compute_batch_false_positive(pred_curr_negative_in_common_positive)
        fp_curr_negative_in_common_positive.append(fp2)
        auc_cond = compute_batch_auc(y_list_cond, pred_list_cond)
        if auc_cond > best_auc:
            best_auc = auc_cond
            torch.save({
                'args': args,
                'model_state_dict': model.state_dict(),
                'best_valid_auc': best_auc
            }, 'best_CondGAE.pt')
        print(f"Step: {step+1}; validation roc_auc (true condition): {auc_cond};")
        print(f"False positive rate: new negative: {fp1}; curr negative in common positive: {fp2}")
        model.train()
      
# import matplotlib.pyplot as plt

# train_idx = [i for i in range(len(train_loss))]
# valid_idx = [i for i in range(len(train_loss)) if (i+1) % 5 == 0 ]

# plt.clf()  
# plt.plot(train_idx, train_loss)
# plt.ylabel("Training loss")
# plt.xlabel("number of batches")
# plt.savefig(f'./training_loss', bbox_inches='tight')
# plt.clf()

# plt.clf()  
# plt.plot(valid_idx, valid_auc)
# plt.ylabel("Validation AUC")
# plt.xlabel("number of batches")
# plt.xticks(valid_idx) 
# plt.savefig(f'./valid_auc', bbox_inches='tight')
# plt.clf()

        

