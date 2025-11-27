from tqdm import tqdm
import numpy as np

import torch
import argparse
from torch import Tensor

from graph_utils.load_seq_event import *
from graph_utils.edge_selection import *

from torch.utils.data import DataLoader

import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

from DM_model.d3pm import DMModel

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--diffusion_steps", type=int, default=1000, help="total number of diffusion steps")
parser.add_argument("--diffusion_schedule", type=str, default="cosine", help="diffusion schedule: linear or cosine")
# backbone GNN
parser.add_argument("--n_layers", type=int, default=12, help="num of GNN layers")
parser.add_argument("--in_channels", type=int, default=32, help="hidden size")
parser.add_argument("--hidden_dim", type=int, default=128, help="hidden size")
parser.add_argument("--aggregation", type=str, default="sum", help="aggregation method")
# parser.add_argument("--p_uncond", type=float, default=0.1, help="CF rate")
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
# parallel sampling
parser.add_argument('--parallel_sampling', type=int, default=1)
parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
parser.add_argument('--inference_schedule', type=str, default="cosine")
args = parser.parse_args()

def compute_batch_auc(y_list, pred_list):
    all_pred = torch.cat(pred_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    all_y, all_pred = all_y.cpu().numpy(), all_pred.cpu().numpy()
    auc = roc_auc_score(all_y, all_pred)
    return auc

def compute_batch_pr_auc(y_list, pred_list):
    all_pred = torch.cat(pred_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    all_y, all_pred = all_y.cpu().numpy(), all_pred.cpu().numpy()
    pr_auc = average_precision_score(all_y, all_pred)
    return pr_auc

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

batch_size=args.batch_size

train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, n = prepare_seq_adj_batch(
    batch_size=batch_size, data_name="amazon")

# feat is shared and fixed for every adj
feat_batch = feat.unsqueeze(0).repeat(batch_size, 1, 1)

dm_model = DMModel(args, device)

optimizer = torch.optim.Adam(
    dm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

dm_model.model.train()

valid_auc = []; valid_pr_auc = []
for epoch in range(10):
    epoch_loss = 0.0
    for step, batch in enumerate(train_dataloader, start=1):
        loss = dm_model.categorical_training_step(batch, feat_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    ###
    print(f"[epoch]: {epoch_loss}; avg_batch_loss: {epoch_loss/step}")
    # eval
    dm_model.model.eval()
    y_list_cond = []; pred_list_cond = []
    for valid_step, batch in enumerate(valid_dataloader, start=1):
        print(f"{valid_step}/{len(valid_dataloader)}", end='\r')
        edge_pred = dm_model.test_set(batch, feat_batch) # [B, N, N] prob of edge=1
        edge_pred = torch.tensor(edge_pred)
        _, output_adj = batch
        y_list_cond.append(output_adj.reshape(-1, 1))
        pred_list_cond.append(edge_pred.reshape(-1, 1))
    ####
    auc_cond = compute_batch_auc(y_list_cond, pred_list_cond)
    valid_auc.append(auc_cond)
    pr_auc_cond = compute_batch_pr_auc(y_list_cond, pred_list_cond)
    valid_pr_auc.append(pr_auc_cond)
    print(f"Epoch: {epoch+1}; validation roc_auc (true condition): {auc_cond}; pr auc: {pr_auc_cond}")
    dm_model.model.train()

validation = {
    'valid_auc': valid_auc, 
    'valid_pr_auc': valid_pr_auc}
torch.save(validation, 'd3pm_validation.pt')

import matplotlib.pyplot as plt

idx = [i for i in range(len(valid_auc))]

plt.clf()  
plt.plot(idx, valid_auc)
plt.ylabel("ROC-AUC")
plt.xlabel("Epoch")
plt.title(f'Best: {max(valid_auc)}')
plt.savefig(f'./validation_roc_auc', bbox_inches='tight')
plt.clf()  


plt.clf()  
plt.plot(idx, valid_pr_auc)
plt.ylabel("Average precision score")
plt.xlabel("Epoch")
plt.title(f'Best: {max(valid_pr_auc)}')
plt.savefig(f'./validation_ap', bbox_inches='tight')
plt.clf()  

# 3. randomize condition (for classifier-free guidance)
# use_uncond = (torch.rand(batch_size, device=device) < p_uncond)
# determine whether use condition for this batch
# need to check why this is useful
# if use_uncond.float().mean() > 0.5:
#     edge_prev_in = None
#     feat_prev_in = None
# else:
#     edge_prev_in = edge_index_list
#     feat_prev_in = feat

# classifier-free guidance in logit space
# w = guidance_scale
# logits_guided = (1 + w) * logits_cond - w * logits_uncond  # [B, N, N, K]
# sample next adjacency from guided distribution

# unconditional logits (no condition)
# logits_uncond = model(A_t, t, edge_prev_in=None, feat_prev_in=None)  # [B, N, N, K]
# conditional logits

