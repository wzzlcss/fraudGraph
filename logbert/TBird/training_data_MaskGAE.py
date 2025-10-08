import numpy as np
import torch
from torch.utils.data import Dataset

def construct_edge(node_seq):
    from_node = []
    to_node = []
    for i in range(len(node_seq)-1):
        from_node.append(node_seq[i])
        to_node.append(node_seq[i+1])
    return np.array([from_node, to_node], dtype=np.int64)

def keep_unique(edge_index):
    # return unique directed edges (np array)
    uniq_pairs, idx = np.unique(edge_index.T, axis=0, return_index=True)
    return uniq_pairs.T

def find_all_edges(all_seqs, vocab_dict):
    # all_seqs = [[vocab_dict[ln] for ln in line.split()] for line in sequence_lists]
    node_seq = np.array(all_seqs[0])
    edge_index = construct_edge(node_seq)
    edge_index = keep_unique(edge_index)
    for idx in range(1, len(all_seqs)):
        node_seq = np.array(all_seqs[idx])
        edge_index_2 = construct_edge(node_seq)
        edge_index_2 = keep_unique(edge_index_2)
        # combine with existing edges
        edge_index = np.hstack([edge_index, edge_index_2])
        edge_index = keep_unique(edge_index)
    return edge_index

def find_all_negative_edges(all_unique_edges, n, exclude_diag=True):
    adj = np.zeros((n, n), dtype=bool)
    u, v = all_unique_edges
    adj[u, v] = True
    if exclude_diag:
        np.fill_diagonal(adj, True)  # mark self-loops as 'existing' to exclude them
    uu, vv = np.where(~adj)
    return np.stack([uu, vv], axis=0)  # (2, M)

def all_edges(n, exclude_diag=True):
    adj = np.zeros((n, n), dtype=bool)
    if exclude_diag:
        np.fill_diagonal(adj, True)  # mark self-loops as 'existing' to exclude them
    uu, vv = np.where(~adj)
    return np.stack([uu, vv], axis=0)  # (2, M)

class LogGraphDataset(Dataset):
    def __init__(self, sequence_lists: list[str], vocab_dict: dict):
        # # each seq can have different length
        # all_seqs = [[vocab_dict[ln] for ln in line.split()] for line in sequence_lists]
        # count = [len(seq) for seq in all_seqs]
        # filtered = [x for x, m in zip(all_seqs, count) if m > 10]
        self.all_seqs = sequence_lists
    
    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx: int):
        node_seq = np.array(self.all_seqs[idx])
        edge_index = construct_edge(node_seq)
        edge_index = keep_unique(edge_index)
        return edge_index
    
    def collate(self, batch):
        # gnn conv is defined for single graph input; always use batch size of 1
        if len(batch) > 1:
            raise RuntimeError("should use a batch size of 1")
        return batch[0]

class LogGraphDatasetAdjPair(Dataset):
    def __init__(self, seq_pair: list[str]):
        # # each seq can have different length
        # all_seqs = [[vocab_dict[ln] for ln in line.split()] for line in sequence_lists]
        # count = [len(seq) for seq in all_seqs]
        # filtered = [x for x, m in zip(all_seqs, count) if m > 10]
        self.seq_pair = seq_pair
    
    def __len__(self):
        return len(self.seq_pair)

    def __getitem__(self, idx: int):
        input_seq, output_seq = self.seq_pair[idx]
        edge_index_input = construct_edge(np.array(input_seq))
        edge_index_input = keep_unique(edge_index_input)
        edge_index_output = construct_edge(np.array(output_seq))
        edge_index_output = keep_unique(edge_index_output)
        return edge_index_input, edge_index_output
    
    def collate(self, batch):
        # gnn conv is defined for single graph pair input; always use batch size of 1
        if len(batch) > 1:
            raise RuntimeError("should use a batch size of 1")
        return batch[0]

# # dataset preprocessing
# import sys
# sys.path.append('../')
# import os
# import pandas as pd
# import numpy as np
# output_dir = "../output/tbird/"
# log_file = "Thunderbird_20M.log"
# df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')
# # data preprocess
# df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
# df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')
# df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
# df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
# df['deltaT'].fillna(0)





