import sys
sys.path.append("../")
sys.path.append("../../")

from tqdm import tqdm
import numpy as np
from bert_pytorch.dataset import WordVocab

import torch
import argparse
from torch import Tensor

from training_data_MaskGAE import keep_unique, construct_edge, LogGraphDatasetAdjPair, find_all_negative_edges, all_edges
from model_MaskGAE import GNNEncoder, EdgeDecoder, CondGAE
from mask_MaskGAE import MaskEdge

from torch.utils.data import DataLoader

import random
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import add_self_loops


train_data_path='../output/tbird/train_with_time'
valid_data_path='../output/tbird/valid_with_time'
vocab_path='../output/tbird/vocab.pkl'
emb_path = '../output/tbird/bert/embedding.pt'

# data_iter[i]: represent a sequence of event id
with open(train_data_path, 'r') as f:
    train_iter = f.readlines()

with open(valid_data_path, 'r') as f:
    valid_iter = f.readlines()

# feat = torch.load(emb_path)
# n_feat = feat.shape[1]

vocab = WordVocab.load_vocab(vocab_path)
print("vocab Size: ", len(vocab))
vocab_dict = vocab.stoi # string-to-index
n = len(vocab) # size of embedding table

def filter_raw_seq_non_pair(data_iter, vocab_dict, min_seq_len):
    eventSeq = []
    for line in data_iter:
        idSeq = []
        et = (line.split()[2]).split(",")
        et_len = len(et)
        # check seq length
        if et_len < min_seq_len:
            continue
        flag = False; idx = 0
        # check whether event is in the vocab
        while (not flag) and (idx < et_len):
            if et[idx] not in vocab_dict:
                flag = True # skip if seq contains new event id
            else:
                idSeq.append(vocab_dict[et[idx]])
                idx += 1
        if flag:
            continue
        else:
            eventSeq.append(idSeq)
    print(f"Total num seq: {len(eventSeq)}")
    # reorder Seq by the starting time stamp
    glist = []
    for seq in eventSeq:
        edge_index = construct_edge(np.array(seq))
        edge_index = keep_unique(edge_index)
        glist.append(edge_index)
    return glist

def all_edges(n, exclude_diag=True):
    adj = np.zeros((n, n), dtype=bool)
    if exclude_diag:
        np.fill_diagonal(adj, True)  # mark self-loops as 'existing' to exclude them
    uu, vv = np.where(~adj)
    return np.stack([uu, vv], axis=0)  # (2, M)

def find_all_edges(glist, n):
    adj = np.zeros((n, n), dtype=int)
    for g in glist:
        for i in range(g.shape[1]):
            u, v = g[0, i], g[1, i]
            adj[u, v] += 1
    return adj

train_g = filter_raw_seq_non_pair(train_iter, vocab_dict, min_seq_len = 10)  
valid_g = filter_raw_seq_non_pair(valid_iter, vocab_dict, min_seq_len = 10) 
train_all_adj = find_all_edges(train_g, n)
valid_all_adj = find_all_edges(valid_g, n)

# out_deg_mx = np.triu(train_all_adj)
# lower_no_diag_mx = np.tril(train_all_adj, k=-1)
# out_deg = out_deg_mx.sum(axis=1)
# in_deg = lower_no_diag_mx.sum(axis=0)
# deg = out_deg + in_deg
# sorted_idx = np.argsort(-deg)
# # select top 50
# selected_idx = sorted_idx[:50]
# selected_idx.sort()


import matplotlib.pyplot as plt

# almost exactly the same adjacency pattern
plt.imshow(train_all_adj[:50, :50], cmap='viridis')
plt.ylabel("Event ID")
plt.xlabel("Event ID")
plt.title(f"System log Tbird: train")
plt.savefig(f'./train.png', bbox_inches='tight')
plt.clf()

plt.imshow(valid_all_adj[:50, :50], cmap='viridis')
plt.ylabel("Event ID")
plt.xlabel("Event ID")
plt.title(f"System log Tbird: validation")
plt.savefig(f'./valid.png', bbox_inches='tight')
plt.clf()

def get_event_freq(data_iter, n):
    adj = np.zeros((n, n), dtype=int)
    adj_exist = np.zeros((n, n), dtype=bool)
    for batch in data_iter:
        in_edge, out_edge = batch
        for i in range(in_edge.shape[1]):
            u, v = in_edge[0, i], in_edge[1, i]
            adj[u, v] += 1; adj[v, u] += 1     
            adj_exist[u, v] = True; adj_exist[v, u] = True        
    # record the remaining
    for i in range(out_edge.shape[1]):
        u, v = out_edge[0, i], out_edge[1, i]
        adj[u, v] += 1; adj[v, u] += 1     
        adj_exist[u, v] = True; adj_exist[v, u] = True 
    return adj, adj_exist

adj_train, adj_train_exist = get_event_freq(train_set, n)
adj_valid, adj_valid_exist = get_event_freq(valid_set, n)
adj_train.sum(1)

# get frequency of event occurrance in training and validation
# get frequency of casuality in training and validation: (% casuality 2 by 2 table)
# get pure BERT results without modeling the casuality graph
# by only modeling the casuality between top events and include more negative edges
# Do check other datasets from the event seq DM paper
