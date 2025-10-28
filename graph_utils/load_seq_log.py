import sys, os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'logbert')))

sys.path.append("../")

from bert_pytorch.dataset import WordVocab
from .training_data_MaskGAE import LogGraphDatasetAdjPair
from .edge_selection import *

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import product

def filter_raw_seq(data_iter, vocab_dict, min_seq_len):
    eventSeq = []
    timeSeq = []
    for line in data_iter:
        idSeq = []; tSeq = []
        et = (line.split()[2]).split(",")
        tt = (line.split()[0]).split(",")
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
                tSeq.append(tt[idx])
                idx += 1
        if flag:
            continue
        else:
            eventSeq.append(idSeq)
            timeSeq.append(tSeq)
    print(f"Total num seq: {len(eventSeq)}")
    start_time = [ln[0] for ln in timeSeq]
    idx = sorted(range(len(start_time)), key=start_time.__getitem__)
    # reorder Seq by the starting time stamp
    eventSeq_sorted_pair = [(eventSeq[idx[i]], eventSeq[idx[i+1]]) for i in range(len(idx)-1)]
    print(f"Total num seq-pair (sorted): {len(eventSeq_sorted_pair)}")
    return eventSeq_sorted_pair, eventSeq, timeSeq

def prepare_seq(name="tbird"):
    # process log sequence dataset
    train_data_path='logbert/output/tbird/train_with_time'
    valid_data_path='logbert/output/tbird/valid_with_time'
    vocab_path='logbert/output/tbird/vocab.pkl'
    emb_path = 'logbert/output/tbird/bert/embedding.pt'
    # data_iter[i]: represent a sequence of event id
    with open(train_data_path, 'r') as f:
        train_iter = f.readlines()[:6000]
    ####
    with open(valid_data_path, 'r') as f:
        valid_iter = f.readlines()[:500]
    ####
    # embedding table trained from BERT used as input features
    feat = torch.load(emb_path)
    # n_feat = feat.shape[1]
    vocab = WordVocab.load_vocab(vocab_path)
    print("vocab Size: ", len(vocab))
    vocab_dict = vocab.stoi # string-to-index
    n = len(vocab) # size of embedding table
    train_data, train_eventSeq, _ = filter_raw_seq(train_iter, vocab_dict, min_seq_len = 10)  
    valid_data, _, _ = filter_raw_seq(valid_iter, vocab_dict, min_seq_len = 10) 
    # dataloader for training gae: g_{t-1}, g_{t} = batch
    train_set = LogGraphDatasetAdjPair(train_data)
    train_dataloader = DataLoader(train_set, batch_size=1, num_workers=1, collate_fn=train_set.collate)
    valid_set = LogGraphDatasetAdjPair(valid_data)
    valid_dataloader = DataLoader(valid_set, batch_size=1, num_workers=1, collate_fn=valid_set.collate)
    return train_dataloader, valid_dataloader, train_eventSeq, feat, n


# # find common positive among training sequence
# common_positive = find_all_edges(train_eventSeq)
# common_positive = remove_self_loop(common_positive)
# common_negative = find_all_negative_edges(
#     common_positive, n, exclude_diag=True)
# common_negative = remove_self_loop(common_negative)

# compare with last time stamp
# given g_{t-1}, g_{t}: 
# return 1. new_positive: positive edges in g_{t} that are negative in g_{t-1}
def find_new_positive(prev_edge, curr_edge):
    prev_tuples = set(map(tuple, prev_edge.T))
    curr_tuples = list(map(tuple, curr_edge.T))
    new_positive = np.array([e for e in curr_tuples if e not in prev_tuples]).T
    return new_positive
# compare with last time stamp
# given g_{t-1}, g_{t}: 
# 2. new_negative: negative edges in g_{t} that are positive in g_{t-1} 
def find_new_negative(prev_edge, curr_edge, n):
    curr_negative = find_all_negative_edges(
        curr_edge, n, exclude_diag=True) # set to true to ignore disappeared self-loop
    curr_negative_tuples = set(map(tuple, curr_negative.T))
    prev_tuples = list(map(tuple, prev_edge.T))
    new_negative = np.array([e for e in prev_tuples if e in curr_negative_tuples]).T
    return new_negative

# compare with common positive set
# given g_{t}, common_positive: 
# return negative edges that are in common positive / not in common positive
def curr_negative_vs_common_positive(curr_edge, common_positive, n):
    common_positive_tuples = set(map(tuple, common_positive.T)) 
    curr_negative = find_all_negative_edges(
        curr_edge, n, exclude_diag=True)
    curr_negative_tuples = list(map(tuple, curr_negative.T))
    curr_negative_in_common_positive = np.array(
        [e for e in curr_negative_tuples if e in common_positive_tuples]).T
    curr_negative_not_in_common_positive = np.array(
        [e for e in curr_negative_tuples if e not in common_positive_tuples]).T
    return curr_negative_in_common_positive, curr_negative_not_in_common_positive
    
# given g_t, negative sampling
def neg_edges_relevant(curr_edge, n):
    # edges not in curr_edge, but with at least one of end points in current graph
    nodes = np.unique(curr_edge)
    edge_set = set(map(tuple, curr_edge.T))
    all_edges = list(product(range(n), range(n)))
    neg_edges = np.array(
        [(u, v) for (u, v) in all_edges if (u, v) not in edge_set and u != v and (u in nodes or v in nodes)]
    ).T
    return neg_edges





