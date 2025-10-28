import numpy as np

def construct_edge(node_seq):
    # given a event sequence, construct causal graph's edges (duplicates not removed)
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

# def edge_end_point

