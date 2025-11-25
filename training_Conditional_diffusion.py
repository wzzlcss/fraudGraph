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

from gae.model_MaskGAE import GNNEncoder
from DM_model.dm import *

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--T", type=int, default=100, help="total number of diffusion steps")
parser.add_argument("--p_uncond", type=float, default=0.1, help="CF rate")
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
# gnn encoder
parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--encoder_dropout', type=float, default=0.3, help='Dropout probability of encoder. (default: 0.7)')
# parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')
args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

batch_size=args.batch_size
T=args.T
p_uncond = args.p_uncond

train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, n = prepare_seq_adj_batch(
    batch_size=batch_size, data_name="amazon")

gammas = make_gamma_schedule(T=T, gamma_start=1e-4, gamma_end=0.2).to(device)

n_feat = feat.shape[1]
# gnn encoder for the cond adj_previous
gnnencoder = GNNEncoder(
    in_channels=n_feat, hidden_channels=args.encoder_channels, out_channels=args.hidden_channels,
    num_layers=args.encoder_layers, dropout=args.encoder_dropout,
    bn=args.bn, layer=args.layer, activation=args.encoder_activation).to(device)

model = GraphDenoiser(d_model=args.hidden_channels, T=T, gnnencoder=gnnencoder).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def training_step(model, batch, feat, gammas, T):
    # p_uncond: probability of dropping condition (classifier-free guidance)
    adj_in, adj_out = batch
    adj_in = adj_in.to(device)
    edge_index_list = adj_to_edge_index_list(adj_in)
    adj_out = adj_out.to(device)
    batch_size = adj_in.shape[0]
    # 1. sample diffusion steps uniformly
    t = torch.randint(1, T + 1, (batch_size, ), device=device, dtype=torch.long)
    # 2. sample noisy adj from q(A_t^noisy | A_0); A_0 = adj_out
    adj_noisy = q_sample_discrete(adj_out, t, gammas)  # [B, N, N]
    # 3) forward pass
    logits_edges = model(adj_noisy, t, edge_prev_in=edge_index_list, feat_prev_in=feat)  # [B, N, N, 2]
    # 4) loss: compute loss by BCE comparing logits with adj_out
    # reshape for CE
    logits_flat = logits_edges.view(batch_size * n * n, 2)           # [B*N*N, 2]
    target_flat = adj_out.view(batch_size * n * n)                   # [B*N*N]
    loss = F.cross_entropy(logits_flat, target_flat)
    return loss

@torch.no_grad()
def sample_next_graph(model, batch, feat, gammas, T):
    # predict adj_out given adj_in, feat using diffusion + CFG
    # guidance_scale: 'w' in CFG: (1+w)*cond - w*uncond
    ###################################
    adj_in, adj_out = batch
    adj_in = adj_in.to(device)
    edge_index_list = adj_to_edge_index_list(adj_in)
    adj_out = adj_out.to(device)
    batch_size = adj_in.shape[0]
    # start from uniform noise adjacency: this is not correct, the model is trained to predict the clean output
    A_t = torch.randint(low=0, high=2, size=(batch_size, n, n), device=device)
    for s in reversed(range(1, T+1)):
        t = torch.full((batch_size,), s, device=device, dtype=torch.long)
        logits = model(A_t, t, edge_prev_in=edge_index_list, feat_prev_in=feat)  # [B, N, N, K]
        probs = F.softmax(logits, dim=-1)  # [B, N, N, K]
        A_t = torch.distributions.Categorical(probs=probs).sample()  # [B, N, N]
    return A_t


model.train()

epoch_loss_list = []
epoch_per_batch_loss_avg = []
for epoch in range(10):
    epoch_loss = 0.0
    for step, batch in enumerate(train_dataloader, start=1):
        loss = training_step(model, batch, feat, gammas, T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss_list.append(epoch_loss)
    epoch_per_batch_loss_avg.append(epoch_loss/step)
    ###
    print(f"[epoch]: {epoch_loss}; avg_batch_loss: {epoch_loss/step}")


# test overfitting one batch
model.train()
batch = next(iter(train_dataloader))
one_batch_loss = []
for it in range(500):
    loss = training_step(model, batch, feat, gammas, T)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    one_batch_loss.append(loss.item())
    print(f'batch loss: {loss.item()}')

training_loss = {
    'train_epoch_loss': epoch_loss_list, 
    'train_epoch_avg_batch_loss': epoch_per_batch_loss_avg,
    'train_one_batch_size10_loss': one_batch_loss}
torch.save(training_loss, 'baseline_dm_training.pt')

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

