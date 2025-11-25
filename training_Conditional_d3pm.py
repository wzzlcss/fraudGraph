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
args = parser.parse_args()

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

epoch_loss_list = []
epoch_per_batch_loss_avg = []

for epoch in range(10):
    epoch_loss = 0.0
    for step, batch in enumerate(train_dataloader, start=1):
        loss = dm_model.categorical_training_step(batch, feat_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss_list.append(epoch_loss)
    epoch_per_batch_loss_avg.append(epoch_loss/step)
    ###
    print(f"[epoch]: {epoch_loss}; avg_batch_loss: {epoch_loss/step}")


# test overfitting one batch
batch = next(iter(train_dataloader))
one_batch_loss = []
for it in range(500):
    loss = dm_model.categorical_training_step(batch, feat_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    one_batch_loss.append(loss.item())
    print(f'batch loss: {loss.item()}')

training_loss = {
    'train_epoch_loss': epoch_loss_list, 
    'train_epoch_avg_batch_loss': epoch_per_batch_loss_avg,
    'train_one_batch_size10_loss': one_batch_loss}
torch.save(training_loss, 'd3pm_training.pt')


import matplotlib.pyplot as plt

plt.clf()  
plt.plot([i for i in range(len(epoch_loss_list))], epoch_loss_list)
plt.ylabel("Epoch Total Loss (On whole training set)")
plt.xlabel("Epoch")
plt.savefig(f'./epoch_loss', bbox_inches='tight')
plt.clf()  


plt.clf()  
plt.plot([i for i in range(len(epoch_per_batch_loss_avg))], epoch_per_batch_loss_avg)
plt.ylabel("Epoch Total Loss / Batch size (On whole training set)")
plt.xlabel("Epoch")
plt.savefig(f'./epoch_loss_avg_batch', bbox_inches='tight')
plt.clf()  


plt.clf()  
plt.plot([i for i in range(len(one_batch_loss))], one_batch_loss, label="")
plt.ylabel("Batch Loss (overfit one batch of size 10)")
plt.xlabel("Iteration")
plt.savefig(f'./loss_one_batch', bbox_inches='tight')
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

