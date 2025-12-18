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

from TGB_baselines.TGAT import TGAT
from TGB_baselines.CAWN import CAWN
from TGB_baselines.TCL import TCL
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
parser.add_argument('--model_name', type=str, default='TGAT', help='name of the model, note that EdgeBank is only applicable for evaluation',
    choices=['TGAT', 'CAWN', 'TCL'])
parser.add_argument('--sample_neighbor_strategy', default='recent', choices=[
    'uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
    'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
    'it works when sample_neighbor_strategy == time_interval_aware')  
parser.add_argument('--num_neighbors', type=int, default=20,  help='number of neighbors to sample for each node')  
parser.add_argument('--time_feat_dim', type=int, default=128, help='dimension of the time embedding')
parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--position_feat_dim', type=int, default=128, help='dimension of the position embedding')
parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
# optimization
parser.add_argument('--optimizer', type=str, default='Adam',
    choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
args = parser.parse_args()

# sequence-level training
def train_on_one_sequence(seq):
    input_edge, output_edge, timeline, edge_ids = seq
    # all_negatives = find_all_negative_edges(
    #     output_edge, n, exclude_diag=True)
    # all_negatives = keep_unique(all_negatives)
    # all_negatives = torch.tensor(all_negatives).to(device).to(dtype=torch.long)
    # DGB assume the node index start from 1
    input_edge += 1; output_edge += 1 #; all_negatives += 1
    # Reset graph / memory for THIS sequence
    # model.reset_state()
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
        raw_node_feature=feat.to(device), 
        raw_edge_feature=torch.zeros(full_seq.shape[1], 1).to(device))
    model[0].set_neighbor_sampler(train_neighbor_sampler)
    # compute loss on the output sequence (teacher forcing)
    # teacher forcing ensures that most recent historical graph is seen at each step
    # enumerate over edges in S2 (every edge has a full access to the past edges)
    src_t, dst_t = output_edge[0], output_edge[1]
    time_t = timeline[input_edge.shape[1]:]
    edge_id = edge_ids[input_edge.shape[1]:]
    # create neg edges from the whole pool
    # neg_size = time_t.shape[0] # sample the same number of negative edges
    # sample_idx = torch.randint(
    #     all_negatives.shape[1], (neg_size,), device=all_negatives.device)
    # neg_samples = all_negatives[:, sample_idx]
    # neg_src, neg_dst = neg_samples[0], neg_samples[1]
    # original implementation: per positive edge get one negative sample and keep the source node the same
    _, neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(src_t))
    neg_src_node_ids = src_t
    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
        # positive pair
        src_node_embeddings, dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_t,
                dst_node_ids=dst_t,
                node_interact_times=time_t, # ths ensures that no feature graph is visible
                num_neighbors=args.num_neighbors)
        # negative pair
        neg_src_node_embeddings, neg_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=neg_src_node_ids,
                dst_node_ids=neg_dst_node_ids,
                node_interact_times=time_t,
                num_neighbors=args.num_neighbors)
    else:
        raise ValueError(
            f"Wrong value for model_name {args.model_name}!")
    # back-propagate
    # get positive and negative probabilities, shape (batch_size, )
    positive_probabilities = model[1](
        input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).sigmoid()
    negative_probabilities = model[1](
        input_1=neg_src_node_embeddings, input_2=neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
    predicts = torch.cat(
        [positive_probabilities, negative_probabilities], dim=0)
    labels = torch.cat(
        [torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
    loss = loss_func(input=predicts, target=labels)
    return loss
    # train_losses.append(loss.item())
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # print(loss)
    # the true history is encoded into the sampler
    # every time_t will only have access to the previous history

def eval_static(seq):
    input_edge, output_edge, timeline, edge_ids = seq
    # define the negative edge pool for computing loss on the output sequence
    all_negatives = find_all_negative_edges(
        output_edge, N, exclude_diag=True)
    all_negatives = keep_unique(all_negatives)
    # DGB assume the node index start from 1
    input_edge += 1; output_edge += 1; all_negatives += 1
    # Reset graph / memory for THIS sequence
    ###################
    # build the graph & memory using the input sequence as known to the model
    cond_len = input_edge.shape[1]
    infer_len = output_edge.shape[1]
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
    model[0].set_feature(
        raw_node_feature=feat.to(device), 
        raw_edge_feature=torch.zeros(cond_len + infer_len, 1).to(device))
    model[0].set_neighbor_sampler(train_neighbor_sampler)
    # test on validation seq
    pos_src, pos_dst = output_edge[0], output_edge[1]
    neg_src, neg_dst = all_negatives[0], all_negatives[1]
    # create time variable to ensure that only graph constructed from S1 is visible
    pos_t = (np.ones(len(pos_src)) * timeline[cond_len]).astype('int64')
    neg_t = (np.ones(len(neg_src)) * timeline[cond_len]).astype('int64')
    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
        # positive pair
        pos_src_embed, pos_dst_embed = \
            model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=pos_src,
                dst_node_ids=pos_dst,
                node_interact_times=pos_t, # ths ensures that no future graph is visible
                num_neighbors=args.num_neighbors)
        # negative pair
        neg_src_embed, neg_dst_embed = \
            model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=neg_src,
                dst_node_ids=neg_dst,
                node_interact_times=neg_t,
                num_neighbors=args.num_neighbors)
    else:
        raise ValueError(
            f"Wrong value for model_name {args.model_name}!")
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

# def eval_roll_out(seq):
#     performance is too bad
#     model.eval()
#     input_edge, output_edge, timeline, edge_ids = seq
#     # DGB assume the node index start from 1
#     input_edge += 1; output_edge += 1 #; all_negatives += 1
#     # Reset graph / memory for THIS sequence
#     ###################
#     # build the graph & memory using the input sequence as known to the model
#     cond_len = input_edge.shape[1]
#     infer_len = output_edge.shape[1]
#     data = pd.DataFrame({
#         "src_node_ids": input_edge[0],
#         "dst_node_ids": input_edge[1],
#         "edge_ids": edge_ids[:cond_len],
#         "node_interact_times": timeline[:cond_len]
#     })
#     train_neighbor_sampler = get_neighbor_sampler(
#         max_node_id=N,
#         data=data, sample_neighbor_strategy=args.sample_neighbor_strategy,
#         time_scaling_factor = args.time_scaling_factor, seed=0)
#     model[0].set_feature(
#         raw_node_feature=feat.to(device), 
#         raw_edge_feature=torch.zeros(cond_len + infer_len, 1).to(device))
#     model[0].set_neighbor_sampler(train_neighbor_sampler)
#     # infer a sequence with the same length of the output sequence
#     # get the score over all possible edges with the src = previous 
#     # then choose the one with the highest prob
#     last_dst = int(output_edge[0][0])
#     infer_src = []
#     infer_dst = []
#     for infer_i in range(infer_len):
#         infer_src.append(last_dst)
#         src_t = (np.ones(N) * last_dst).astype(input_edge.dtype)
#         dst_t = np.arange(1, N+1) 
#         time_t = (np.ones(N) *  timeline[cond_len + infer_i]).astype(input_edge.dtype)
#         # for idx in range(infer_len):
#         # if args.model_name in ['TGAT', 'CAWN', 'TCL']:
#         src_node_embeddings, dst_node_embeddings = \
#             model[0].compute_src_dst_node_temporal_embeddings(
#                 src_node_ids=src_t,
#                 dst_node_ids=dst_t,
#                 node_interact_times=time_t, # ths ensures that no feature graph is visible
#                 num_neighbors=args.num_neighbors)
#         prob = model[1](
#             input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).sigmoid()
#         # select the predicted dst node that has the highest prob
#         pred_dst_idx = prob.argmax() 
#         pred_dst = int(dst_t[pred_dst_idx])
#         infer_dst.append(pred_dst)
#         last_dst = pred_dst
#         # update memory with inferred nodes
#         data = pd.DataFrame({
#             "src_node_ids": np.append(input_edge[0], infer_src),
#             "dst_node_ids": np.append(input_edge[1], infer_dst),
#             "edge_ids": edge_ids[:cond_len + infer_i + 1], # dummy edge feature
#             "node_interact_times": timeline[:cond_len + infer_i + 1]
#         })
#         train_neighbor_sampler = get_neighbor_sampler(
#             max_node_id=N,
#             data=data, sample_neighbor_strategy=args.sample_neighbor_strategy,
#             time_scaling_factor = args.time_scaling_factor, seed=0)
#         model[0].set_neighbor_sampler(train_neighbor_sampler)

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

train_dataloader, valid_dataloader, test_dataloader, feat, N = prepare_seq_temporal(data_name=args.dataset_name)
# train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, N = prepare_seq_temporal(
#     data_name=args.dataset_name)

# DGB assume the node index start from 1; add a dummy row to ensure correct node feat is used
dummy_row = torch.zeros(1, feat.shape[1], device=feat.device, dtype=feat.dtype)
feat = torch.cat([dummy_row, feat], dim=0)

# model parameters are shared across sequence but graph memory is maintained per sequence
if args.model_name == 'TGAT':
    dynamic_backbone = TGAT(
        node_raw_features_dim=feat.shape[1], 
        edge_raw_features_dim=1, # no edge features
        time_feat_dim=args.time_feat_dim,
        num_heads=args.num_heads, dropout=args.dropout, device=device)
elif args.model_name == 'CAWN':
    dynamic_backbone = CAWN(
        node_raw_features_dim=feat.shape[1], 
        edge_raw_features_dim=1, 
        # neighbor_sampler=train_neighbor_sampler,
        time_feat_dim=args.time_feat_dim, 
        position_feat_dim=args.position_feat_dim, 
        walk_length=args.walk_length,
        num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=device)
elif args.model_name == 'TCL':
    dynamic_backbone = TCL(
        node_raw_features_dim=feat.shape[1], 
        edge_raw_features_dim=1, 
        # neighbor_sampler=train_neighbor_sampler,
        time_feat_dim=args.time_feat_dim, 
        num_layers=args.num_layers, num_heads=args.num_heads,
        num_depths=args.num_neighbors + 1, dropout=args.dropout, device=device)
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

res = {
    "args": args, 
    "train_loss": train_loss, 
    "valid_auc_list": valid_auc_list,
    "valid_pa": valid_pa,
    "test_auc_list": test_auc_list,
    "test_pa": test_pa}

filename = f"{output_path}/{args.dataset_name}_{args.model_name}.pt"
torch.save(res, filename)



