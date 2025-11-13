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
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader, # this maintain an undirected graph
)

from TGN_model.tgn_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--memory_dim', type=int, default=100, help='size of memory embedding in TGN')
parser.add_argument('--time_dim', type=int, default=100, help='size of time embedding in TGN')
parser.add_argument('--embedding_dim', type=int, default=100, help='size of embedding in TGN')
parser.add_argument('--tgn_batch_size', type=int, default=5, help='how many events are processed each step in TGN')
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

def find_nontrivial_negative(input_edge, output_edge, n, common_positive):
    # new negative compared with t-1
    new_negative = find_new_negative(input_edge, output_edge, n)
    # curr negative that is in common positive
    curr_negative_in_common_positive, _ = curr_negative_vs_common_positive(
        output_edge, common_positive, n)
    # negative that is relevant to the current graph
    # relevant_negative = neg_edges_relevant(output_edge, n)
    return new_negative, curr_negative_in_common_positive #, relevant_negative

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

def batch_iteration_training(batch):
    # batch size = 1
    input_edge, output_edge = batch
    input_size = input_edge.shape[1]
    # define the negative edge pool for computing loss on the output sequence
    all_negatives = find_all_negative_edges(
        output_edge, n, exclude_diag=True)
    all_negatives = keep_unique(all_negatives)
    all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
    ##
    input_edge = torch.tensor(input_edge).to(device)
    output_edge = torch.tensor(output_edge).to(device)
    # assign position as time for the whole sequence
    t = torch.arange(input_edge.shape[1] + output_edge.shape[1], device=device)
    # process the input sequence to update the hidden state
    for start in range(0, input_size, tgn_batch_size):
        end = min(start + tgn_batch_size, input_size)
        edge_batch = input_edge[:, start:end]
        t_batch = t[start:end]
        # get all nodes involved in this batch
        src, pos_dst = edge_batch[0], edge_batch[1]
        # n_id = torch.cat([src, pos_dst]).unique()
        dummy_msg = torch.zeros((src.size(0), 1), device=src.device)
        memory.update_state(src, pos_dst, t_batch, dummy_msg)
        neighbor_loader.insert(src, pos_dst)
    # ==========
    # train on the output sequence
    pred_size = output_edge.shape[1]
    # to compare with method that computes loss on all negative samples
    neg_size_unit = int(all_negatives.shape[1] / pred_size)
    # process output edges in order
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(n, dtype=torch.long, device=device)
    loss = 0.0
    for idx in range(pred_size):
        t_pred = t[idx + input_size]
        pos_src, pos_dst = output_edge[0, idx:idx+1], output_edge[1, idx:idx+1]
        # sample negative edges
        sample_idx = torch.randint(
            all_negatives.shape[1], (neg_size_unit,), device=all_negatives.device)
        neg_samples = all_negatives[:, sample_idx]
        neg_src, neg_dst = neg_samples[0], neg_samples[1]
        n_id = torch.cat([pos_src, pos_dst, neg_src, neg_dst]).unique()
        ###
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        dummy_msg = torch.zeros((edge_index.size(1), 1), device=src.device)
        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, t[e_id].to(device), dummy_msg)
        # compute loss
        pos_out = link_pred(z[assoc[pos_src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[neg_src]], z[assoc[neg_dst]])
        # Update memory and neighbor loader with ground-truth state. (in training)
        loss += criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
        dummy_msg_curr = torch.zeros((1, 1), device=src.device)
        memory.update_state(pos_src, pos_dst, t[idx:idx+1], dummy_msg_curr)
        neighbor_loader.insert(pos_src, pos_dst)
    return loss


def batch_iteration_eval(batch):
    # batch size = 1
    input_edge, output_edge = batch
    input_size = input_edge.shape[1]
    # define the negative edge pool for computing loss on the output sequence
    all_negatives = find_all_negative_edges(
        output_edge, n, exclude_diag=True)
    all_negatives = keep_unique(all_negatives)
    all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
    ##
    input_edge = torch.tensor(input_edge).to(device)
    output_edge = torch.tensor(output_edge).to(device)
    # assign position as time for the whole sequence
    t = torch.arange(input_edge.shape[1] + output_edge.shape[1], device=device)
    # process the input sequence to update the hidden state
    for start in range(0, input_size, tgn_batch_size):
        end = min(start + tgn_batch_size, input_size)
        edge_batch = input_edge[:, start:end]
        t_batch = t[start:end]
        # get all nodes involved in this batch
        src, pos_dst = edge_batch[0], edge_batch[1]
        # n_id = torch.cat([src, pos_dst]).unique()
        dummy_msg = torch.zeros((src.size(0), 1), device=src.device)
        memory.update_state(src, pos_dst, t_batch, dummy_msg)
        neighbor_loader.insert(src, pos_dst)
    # ==========
    # train on the output sequence
    pred_size = output_edge.shape[1]
    # to compare with method that computes loss on all negative samples
    neg_size_unit = int(all_negatives.shape[1] / pred_size)
    # process output edges in order
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(n, dtype=torch.long, device=device)
    loss = 0.0
    y_list_cond = []; pred_list_cond = []
    for idx in range(pred_size):
        t_pred = t[idx + input_size]
        pos_src, pos_dst = output_edge[0, idx:idx+1], output_edge[1, idx:idx+1]
        # sample negative edges
        sample_idx = torch.randint(
            all_negatives.shape[1], (neg_size_unit,), device=all_negatives.device)
        neg_samples = all_negatives[:, sample_idx]
        neg_src, neg_dst = neg_samples[0], neg_samples[1]
        n_id = torch.cat([pos_src, pos_dst, neg_src, neg_dst]).unique()
        ###
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        dummy_msg = torch.zeros((edge_index.size(1), 1), device=src.device)
        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, t[e_id].to(device), dummy_msg)
        # compute loss
        pos_out = link_pred(z[assoc[pos_src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[neg_src]], z[assoc[neg_dst]])
        pred = torch.cat([pos_out, neg_out], dim=0)
        pos_y = pos_out.new_ones(pos_out.size(0))
        neg_y = neg_out.new_zeros(neg_out.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y_list_cond.append(y.detach().cpu()); pred_list_cond.append(pred.squeeze().detach().cpu())
    return y_list_cond, pred_list_cond

memory_dim = args.memory_dim
time_dim = args.time_dim
embedding_dim = args.embedding_dim
tgn_batch_size = args.tgn_batch_size

train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, n = prepare_seq(
    data_name="amazon")

# find common positive among training sequence
common_positive = find_all_edges(train_eventSeq)
common_positive = remove_self_loop(common_positive)
common_negative = find_all_negative_edges(
    common_positive, n, exclude_diag=True)
common_negative = remove_self_loop(common_negative)

n_feat = feat.shape[1]


# TGN module
memory = TGNMemory(
    num_nodes=n,
    raw_msg_dim=1, # no msg feature
    memory_dim=memory_dim,
    time_dim=time_dim,
    message_module=IdentityMessage(1, memory_dim, time_dim),
    aggregator_module=LastAggregator(), 
    # for any node that has multiple queued messages, the aggregator picks the most recent one
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=1,
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

neighbor_loader = LastNeighborLoader(n, size=10, device=device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

nepochs = 10

accum_steps = 1000
n_train_sample = len(train_dataloader)
remainder = n_train_sample % accum_steps
last_chunk_size = remainder if remainder != 0 else acc_steps
last_chunk_step = n_train_sample - remainder

memory.train()
gnn.train()
link_pred.train()

train_loss = []
valid_auc = []
valid_pr_auc = []

# training
for epoch in range(nepochs):
    running_loss = 0.0
    for step, batch in enumerate(train_dataloader, start=1):
        print(step, end='\r')
        # edges = np.concatenate((input_edge, output_edge), axis=1)
        # process input edge in order; 
        # reset memory for each pair of input (historical seq) - output (future seq)
        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.
        batch_loss = batch_iteration_training(batch)
        # define training norm
        is_acc_step = ((step + 1) % accum_steps == 0)
        is_last_chunk_step = ((step + 1) >= last_chunk_step)
        demon = accum_steps if not is_last_chunk_step else remainder
        is_last_step = ((n + 1) == n_train_sample)
        ####
        batch_loss = batch_loss / demon
        running_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        memory.detach()
        # print(f"Total avg Loss ({step+1} / {n_train_sample}): {batch_loss.item()}")
        ####
        if is_acc_step or is_last_step: # make change here
            train_loss.append(running_loss)
            print(f"Total avg Loss ({step+1} / {n_train_sample}): {running_loss}")
            running_loss = 0.0
            # evaluation
            memory.eval()
            gnn.eval()
            link_pred.eval()
            y_list_cond = []; pred_list_cond = []
            for _, batch in enumerate(valid_dataloader, start=1):
                print(_, end='\r')
                memory.reset_state()  # Start with a fresh memory.
                neighbor_loader.reset_state()  # Start with an empty graph.
                y_list_cond_i, pred_list_cond_i = batch_iteration_eval(batch)
                y_list_cond += y_list_cond_i
                pred_list_cond += pred_list_cond_i
            ####
            auc_cond = compute_batch_auc(y_list_cond, pred_list_cond)
            valid_auc.append(auc_cond)
            pr_auc_cond = compute_batch_pr_auc(y_list_cond, pred_list_cond)
            valid_pr_auc.append(pr_auc_cond)
            print(f"Step: {step+1}; validation roc_auc (true condition): {auc_cond}; pr auc: {pr_auc_cond}")
            memory.train()
            gnn.train()
            link_pred.train()




    

# within a batch, egde is ordered by time
# src: source node [batch_size]
# pos_dst: positive dst node (edge exist) [batch_size]
# for each edge, sample one negative dst
# neg_dst [batch_size]
# n_id: all nodes involved in src, pos_dst, neg_dst
# pos_out, neg_out = model(
#     batch=batch, n_id=n_id, msg=data.msg[e_id].to(device), t=data.t[e_id].to(device),
#     edge_index=edge_index, id_mapper=helper, src_indic=src_indic)



# input_edge, output_edge = batch
# edges = np.concatenate((input_edge, output_edge), axis=1)
# # process input edge in order; 
# # reset memory for each pair of input (historical seq) - output (future seq)
# memory.train()
# gnn.train()
# link_pred.train()
# memory.reset_state()  # Start with a fresh memory.
# neighbor_loader.reset_state()  # Start with an empty graph.
# num_edges = edges.shape[1]
# new_negative, curr_negative_in_common_positive = find_nontrivial_negative(
#     input_edge, output_edge, n, common_positive)
# # assign position as time for the whole sequence
# t = torch.arange(num_edges, device=device)
# for start in range(0, num_edges, tgn_batch_size):
#     end = min(start + tgn_batch_size, num_edges-1)
#     edge_batch = edges[:, start:end]
#     t_batch = t[start:end]
#     # get all nodes involved in this batch
#     src, pos_dst = edge_batch[0].tolist(), edge_batch[1].tolist()
#     # sample negative dist

#     n_id = list(set(src + pos_dst))
#     n_id = torch.tensor(n_id).to(dtype=torch.int64).to(device)
#     # get the graph that have been seen at this step
#     n_id, edge_index, e_id = neighbor_loader(n_id)
#     assoc[n_id] = torch.arange(n_id.size(0), device=device)
#     # # Get updated memory of all nodes involved in the computation.
#     # z, last_update = memory(n_id)
#     # z = gnn(z, last_update, edge_index, t_batch)
#     src = torch.tensor(src).to(device)
#     pos_dst = torch.tensor(pos_dst).to(device)
#     dummy_msg = torch.zeros((src.size(0), 0), device=src.device)
#     memory.update_state(src, pos_dst, t_batch, dummy_msg)
#     neighbor_loader.insert(src, dst)


# # training
# for epoch in range(args.epochs):
#     for step, batch in enumerate(train_dataloader, start=1):
#         input_edge, output_edge = batch
#         # edges = np.concatenate((input_edge, output_edge), axis=1)
#         # process input edge in order; 
#         # reset memory for each pair of input (historical seq) - output (future seq)
#         memory.train()
#         gnn.train()
#         link_pred.train()
#         memory.reset_state()  # Start with a fresh memory.
#         neighbor_loader.reset_state()  # Start with an empty graph.
#         input_size = input_edge.shape[1]
#         # define the negative edge pool for computing loss on the output sequence
#         all_negatives = find_all_negative_edges(
#             output_edge, n, exclude_diag=True)
#         all_negatives = keep_unique(all_negatives)
#         all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
#         ##
#         input_edge = torch.tensor(input_edge).to(device)
#         output_edge = torch.tensor(output_edge).to(device)
#         # assign position as time for the whole sequence
#         t = torch.arange(input_edge.shape[1] + output_edge.shape[1], device=device)
#         # process the input sequence to update the hidden state
#         for start in range(0, input_size, tgn_batch_size):
#             end = min(start + tgn_batch_size, input_size)
#             edge_batch = input_edge[:, start:end]
#             t_batch = t[start:end]
#             # get all nodes involved in this batch
#             src, pos_dst = edge_batch[0], edge_batch[1]
#             # n_id = torch.cat([src, pos_dst]).unique()
#             dummy_msg = torch.zeros((src.size(0), 1), device=src.device)
#             memory.update_state(src, pos_dst, t_batch, dummy_msg)
#             neighbor_loader.insert(src, pos_dst)
#         # ==========
#         # train on the output sequence
#         pred_size = output_edge.shape[1]
#         # to compare with method that computes loss on all negative samples
#         neg_size_unit = int(all_negatives.shape[1] / pred_size)
#         # process output edges in order
#         # Helper vector to map global node indices to local ones.
#         assoc = torch.empty(n, dtype=torch.long, device=device)
#         for idx in range(pred_size):
#             t_pred = t[idx + input_size]
#             pos_src, pos_dst = output_edge[0, idx:idx+1], output_edge[1, idx:idx+1]
#             # sample negative edges
#             sample_idx = torch.randint(
#                 all_negatives.shape[1], (neg_size_unit,), device=all_negatives.device)
#             neg_samples = all_negatives[:, sample_idx]
#             neg_src, neg_dst = neg_samples[0], neg_samples[1]
#             n_id = torch.cat([pos_src, pos_dst, neg_src, neg_dst]).unique()
#             ###
#             n_id, edge_index, e_id = neighbor_loader(n_id)
#             assoc[n_id] = torch.arange(n_id.size(0), device=device)
#             dummy_msg = torch.zeros((edge_index.size(1), 1), device=src.device)
#             # Get updated memory of all nodes involved in the computation.
#             z, last_update = memory(n_id)
#             z = gnn(z, last_update, edge_index, t[e_id].to(device), dummy_msg)
#             # compute loss
#             pos_out = link_pred(z[assoc[pos_src]], z[assoc[pos_dst]])
#             neg_out = link_pred(z[assoc[neg_src]], z[assoc[neg_dst]])
#             loss = criterion(pos_out, torch.ones_like(pos_out))
#             loss += criterion(neg_out, torch.zeros_like(neg_out))
#             # Update memory and neighbor loader with ground-truth state. (in training)
#             dummy_msg_curr = torch.zeros((1, 1), device=src.device)
#             memory.update_state(pos_src, pos_dst, t[idx:idx+1], dummy_msg_curr)
#             neighbor_loader.insert(pos_src, pos_dst)
#             ####
#             loss.backward()
#             optimizer.step()
#             memory.detach()
#             ####