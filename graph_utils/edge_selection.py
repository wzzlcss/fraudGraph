import numpy as np
import torch

def construct_edge(node_seq):
    # given a event sequence, construct causal graph's edges (duplicates not removed)
    from_node = []
    to_node = []
    for i in range(len(node_seq)-1):
        from_node.append(node_seq[i])
        to_node.append(node_seq[i+1])
    return np.array([from_node, to_node], dtype=np.int64)

def construct_adj(node_seq, n):
    # given a event sequence, construct causal graph's adjacency (only indicates existence; diagonal is included)
    adj = np.zeros((n, n))
    for i in range(len(node_seq)-1):
        adj[node_seq[i], node_seq[i+1]] = 1
    return adj

def adj_to_edge_index_list(adj_b):
    # adj: batch of adj tensor
    # find all i, j such that adj[i, j] != 0
    edge_index_list = []
    B = adj_b.shape[0]
    for i in range(B):
        src, dst = torch.nonzero(adj_b[i], as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0).long()
        edge_index_list.append(edge_index)
    return edge_index_list

def keep_unique(edge_index):
    # return unique directed edges (np array)
    uniq_pairs, idx = np.unique(edge_index.T, axis=0, return_index=True)
    return uniq_pairs.T

def remove_self_loop(edge_index):
    mask = edge_index[0] != edge_index[1]
    edge_list_no_selfloop = edge_index[:, mask]
    return edge_list_no_selfloop

def find_all_edges(all_seqs):
    # given all event sequence, find all positive edges that has ever existed
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
    # find the complement of all_unnique_edges
    # exclude_diag = True to exclude diagonals in the returned results
    adj = np.zeros((n, n), dtype=bool)
    u, v = all_unique_edges
    adj[u, v] = True
    if exclude_diag:
        np.fill_diagonal(adj, True)  # mark self-loops as 'existing' to exclude them
    uu, vv = np.where(~adj)
    return np.stack([uu, vv], axis=0)  # (2, M)

def all_edges(n, exclude_diag=True):
    # return all possible edges
    # exclude_diag = True to exclude diagonals in the returned results
    adj = np.zeros((n, n), dtype=bool)
    if exclude_diag:
        np.fill_diagonal(adj, True)  # mark self-loops as 'existing' to exclude them
    uu, vv = np.where(~adj)
    return np.stack([uu, vv], axis=0)  # (2, M)

def trival_negative(all_seqs, n):
    # all_seqs: list of event sequences
    # n: event ids are 0, ..., n-1
    all_unique_edges = find_all_edges(all_seqs)
    all_negative_edges = find_all_negative_edges(all_unique_edges, n, exclude_diag=True)
    all_negative_edges = torch.tensor(all_negative_edges).to(device).to(dtype=torch.long)
    return all_negative_edges

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


