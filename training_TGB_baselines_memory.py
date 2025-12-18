import argparse
from graph_utils.load_seq_event import *
from graph_utils.edge_selection import *
from torch.utils.data import DataLoader
import random
from utils import compute_batch_auc, compute_batch_pr_auc

import logging
import timeit
import time
import datetime
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import os.path as osp
import pandas as pd

from TGB_baselines.MemoryModel import MemoryModel
from TGB_baselines.utils.utils import convert_to_gpu, create_optimizer
from TGB_baselines.utils.utils import get_neighbor_sampler, NegativeEdgeSampler_local
from TGB_baselines.modules import MergeLayer
from utils import compute_batch_auc, compute_batch_pr_auc

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='amazon')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')
parser.add_argument("--batch_size", type=int, default=200)
# model parameters
parser.add_argument('--model_name', type=str, default='TGN', help='name of the model, note that EdgeBank is only applicable for evaluation',
    choices=['DyRep', 'TGN'])
parser.add_argument('--sample_neighbor_strategy', default='recent', choices=[
    'uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
    'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
    'it works when sample_neighbor_strategy == time_interval_aware')  
parser.add_argument('--num_neighbors', type=int, default=20,  help='number of neighbors to sample for each node')  
parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
parser.add_argument('--num_heads', type=int, default=1, help='number of heads used in attention layer')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
# optimization
parser.add_argument('--optimizer', type=str, default='Adam',
    choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
args = parser.parse_args()

# sequence-level training
def train_on_one_sequence(seq):
    input_edge, output_edge, timeline, edge_ids = seq
    cond_len = input_edge.shape[1]
    # DGB assume the node index start from 1
    input_edge += 1; output_edge += 1 #; all_negatives += 1
    # Reset graph / memory for THIS sequence
    ###################
    # build the graph & memory using the input sequence as known to the model
    full_seq = np.concatenate((input_edge, output_edge), axis=1)
    data = pd.DataFrame({
        "src_node_ids": full_seq[0],
        "dst_node_ids": full_seq[1],
        "edge_ids": edge_ids,
        "node_interact_times": timeline
    })
    train_neighbor_sampler = get_neighbor_sampler(
        max_node_id=N,
        data=data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor = args.time_scaling_factor, seed=0)
    # initialize negative samplers
    train_neg_edge_sampler = NegativeEdgeSampler_local(
        src_node_ids = data.src_node_ids, dst_node_ids = data.dst_node_ids)
    model[0].set_feature(
        node_raw_features=feat.to(device), 
        edge_raw_features=torch.zeros(full_seq.shape[1], 1).to(device))
    # reset memory and sampler (which knows graph) for each sequence
    model[0].set_neighbor_sampler(train_neighbor_sampler)
    model[0].memory_bank.__init_memory_bank__()
    # use input sequence to set up memory
    for (u, v, t, eid) in zip(input_edge[0], input_edge[1], timeline[:cond_len], edge_ids[:cond_len]):
        _, _ = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=np.array([u]), 
            dst_node_ids=np.array([v]),
            node_interact_times=np.array([t], dtype=np.float64),
            edge_ids=np.array([eid]), 
            edges_are_positive=True,
            num_neighbors=args.num_neighbors)
    # compute loss on the output sequence (teacher forcing)
    # teacher forcing ensures that most recent historical graph is seen at each step
    # enumerate over edges in S2 (every edge has a full access to the past edges)
    src_t, dst_t = output_edge[0], output_edge[1]
    time_t = timeline[cond_len:]
    edge_id = edge_ids[cond_len:]
    neg_src_emb_list = []; neg_dst_emb_list = []
    pos_src_emb_list = []; pos_dst_emb_list = []
    for (src, dst, t, eid) in zip(src_t, dst_t, time_t, edge_id):
        # original implementation: per positive edge get one negative sample and keep the source node the same
        _, neg_dst = train_neg_edge_sampler.sample(size=1)
        neg_src = np.array([src])
        t1 = np.array([t], dtype=np.int64)
        # note that negative nodes do not change the memories while the positive nodes change the memories,
        # we need to first compute the embeddings of negative nodes for memory-based models
        # get temporal embedding of negative source and negative destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        neg_src_node_embeddings, neg_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=neg_src,
                dst_node_ids=neg_dst,
                node_interact_times=t1,
                edge_ids=None,
                edges_are_positive=False,
                num_neighbors=args.num_neighbors)
        neg_src_emb_list.append(neg_src_node_embeddings)
        neg_dst_emb_list.append(neg_dst_node_embeddings)
        ## postive 
        src_node_embeddings, dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=np.array([src]),
                dst_node_ids=np.array([dst]),
                node_interact_times=t1,
                edge_ids=np.array([eid]),
                edges_are_positive=True,
                num_neighbors=args.num_neighbors)
        pos_src_emb_list.append(src_node_embeddings)
        pos_dst_emb_list.append(dst_node_embeddings)
    # back-propagate
    # get positive and negative probabilities, shape (batch_size, )
    positive_probabilities = model[1](
        input_1=torch.cat(pos_src_emb_list, dim=0), 
        input_2=torch.cat(pos_dst_emb_list, dim=0)).squeeze(dim=-1).sigmoid()
    negative_probabilities = model[1](
        input_1=torch.cat(neg_src_emb_list, dim=0), 
        input_2=torch.cat(neg_dst_emb_list, dim=0)).squeeze(dim=-1).sigmoid()
    predicts = torch.cat(
        [positive_probabilities, negative_probabilities], dim=0)
    labels = torch.cat(
        [torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
    loss = loss_func(input=predicts, target=labels)
    return loss

def eval_static(seq):
    input_edge, output_edge, timeline, edge_ids = seq
    cond_len = input_edge.shape[1]
    infer_len = output_edge.shape[1]
    # define the negative edge pool for computing loss on the output sequence
    all_negatives = find_all_negative_edges(
        output_edge, N, exclude_diag=True)
    all_negatives = keep_unique(all_negatives)
    # DGB assume the node index start from 1
    input_edge += 1; output_edge += 1; all_negatives += 1
    # Reset graph / memory for THIS sequence
    ###################
    # build the graph & memory using the input sequence as known to the model
    data = pd.DataFrame({
        "src_node_ids": input_edge[0],
        "dst_node_ids": input_edge[1],
        "edge_ids": edge_ids[:cond_len],
        "node_interact_times": timeline[:cond_len]
    })
    train_neighbor_sampler = get_neighbor_sampler(
        max_node_id=N, 
        data=data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor = args.time_scaling_factor, seed=0)
    # initialize negative samplers
    train_neg_edge_sampler = NegativeEdgeSampler_local(
        src_node_ids = data.src_node_ids, dst_node_ids = data.dst_node_ids)
    model[0].set_feature(
        node_raw_features=feat.to(device), 
        edge_raw_features=torch.zeros(cond_len + infer_len, 1).to(device))
    # reset memory and sampler (which knows graph) for each sequence
    model[0].set_neighbor_sampler(train_neighbor_sampler)
    model[0].memory_bank.__init_memory_bank__()
    # use input sequence to set up memory
    for (u, v, t, eid) in zip(input_edge[0], input_edge[1], timeline[:cond_len], edge_ids[:cond_len]):
        _, _ = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=np.array([u]), 
            dst_node_ids=np.array([v]),
            node_interact_times=np.array([t], dtype=np.float64),
            edge_ids=np.array([eid]), 
            edges_are_positive=True,
            num_neighbors=args.num_neighbors)
    # test on validation seq
    pos_src, pos_dst = output_edge[0], output_edge[1]
    neg_src, neg_dst = all_negatives[0], all_negatives[1]
    # create time variable to ensure that only graph constructed from S1 is visible
    pos_t = (np.ones(len(pos_src)) * timeline[cond_len]).astype('int64')
    neg_t = (np.ones(len(neg_src)) * timeline[cond_len]).astype('int64')
    neg_src_embed, neg_dst_embed = \
        model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=neg_src,
            dst_node_ids=neg_dst,
            node_interact_times=neg_t,
            edge_ids=None,
            edges_are_positive=False,
            num_neighbors=args.num_neighbors)
    # positive embedding (NO memory update)
    pos_src_embed, pos_dst_embed = \
        model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=pos_src,
            dst_node_ids=pos_dst,
            node_interact_times=pos_t,
            edge_ids=None,
            edges_are_positive=False,
            num_neighbors=args.num_neighbors)
    # get positive and negative probabilities, shape (batch_size, )
    pos_prob = model[1](
        input_1=pos_src_embed, input_2=pos_dst_embed).squeeze(dim=-1).sigmoid()
    neg_prob = model[1](
        input_1=neg_src_embed, input_2=neg_dst_embed).squeeze(dim=-1).sigmoid()
    predicts = torch.cat(
        [pos_prob, neg_prob], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0)
    return predicts.detach().cpu(), labels.cpu()


device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, N = prepare_seq_temporal(
    data_name=args.dataset_name)

# DGB assume the node index start from 1; add a dummy row to ensure correct node feat is used
dummy_row = torch.zeros(1, feat.shape[1], device=feat.device, dtype=feat.dtype)
feat = torch.cat([dummy_row, feat], dim=0)

# model parameters are shared across sequence but graph memory is maintained per sequence
if args.model_name in ['DyRep', 'TGN']:
    dynamic_backbone = MemoryModel(
        node_raw_features_dim=feat.shape[1], 
        edge_raw_features_dim=1, 
        num_nodes=N+1, # include the padding node
        time_feat_dim=args.time_feat_dim,
        model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout,  
        device=args.device)
else:
    raise ValueError(f"Wrong value for model_name {args.model_name}!")

link_predictor = MergeLayer(
    input_dim1=feat.shape[1], 
    input_dim2=feat.shape[1],
    hidden_dim=feat.shape[1], 
    output_dim=1)

model = nn.Sequential(dynamic_backbone, link_predictor)

# define optimizer
optimizer = create_optimizer(
    model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

model = convert_to_gpu(model, device=device)

loss_func = nn.BCELoss()

n_train_sample = len(train_dataloader)
assert n_train_sample % args.batch_size == 0, \
    f"train_dataloader length ({n_train_sample}) must be divisible by batch_size ({args.batch_size})"

# outer loop: Each sequence is independent.
# perform back-propagation once batch_size num of seq is processed
train_loss = []
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, train_seq in enumerate(train_dataloader, start=1):
        print(step, end='\r')
        loss = train_on_one_sequence(train_seq)
                # define training norm
        loss = loss / args.batch_size
        loss.backward()
        model[0].memory_bank.detach_memory_bank()
        running_loss += loss.item()
        # train_losses.append(loss.item())
        if step % args.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            running_loss = running_loss / args.batch_size # avg batch loss
            train_loss.append(running_loss)
            print(f"Avg Batch Loss ({step} / {n_train_sample}): {running_loss}")
            running_loss = 0.0
    # validation after each epoch
    model.eval()
    y_list_cond = []; pred_list_cond = []
    for _, valid_seq in enumerate(valid_dataloader, start=1):
        print(_, end='\r')
        predicts, labels = eval_static(valid_seq)
        y_list_cond.append(labels)
        pred_list_cond.append(predicts)
    ####
    auc_cond = compute_batch_auc(y_list_cond, pred_list_cond)
    pr_auc_cond = compute_batch_pr_auc(y_list_cond, pred_list_cond)
    print(f"Step: {epoch+1}; validation roc_auc (true condition): {auc_cond}; pr auc: {pr_auc_cond}")
