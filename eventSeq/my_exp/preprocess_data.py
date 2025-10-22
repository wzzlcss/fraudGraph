import pickle
import torch
import numpy as np

import matplotlib.pyplot as plt

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

def load_event(data_name, mode):
    dataset_dir = f"../data/{data_name}/{mode}.pkl"
    with open(dataset_dir, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[mode]
    # time_seq = [[x["time_since_start"] for x in seq] for seq in data]
    # time_seq = [torch.tensor(seq[1:]) for seq in time_seq]
    event_seq = [[x["type_event"] for x in seq] for seq in data]
    event_seq = [torch.tensor(seq[1:]) for seq in event_seq]
    return event_seq, num_types

def build_causal_graph(eventSeq):
    glist = []
    for seq in eventSeq:
        edge_index = construct_edge(np.array(seq.numpy()))
        edge_index = keep_unique(edge_index)
        glist.append(edge_index)
    return glist

def find_all_edges(glist, n):
    adj = np.zeros((n, n), dtype=int)
    for g in glist:
        for i in range(g.shape[1]):
            u, v = g[0, i], g[1, i]
            adj[u, v] += 1
    return adj


data_name = "amazon"
mode = "train"

def plot_causal_graph(data_name, mode):
    data, n = load_event(data_name, mode)
    g = build_causal_graph(data)
    all_adj = find_all_edges(g, n)
    plt.clf()  
    plt.imshow(all_adj, cmap='viridis')
    plt.ylabel("Event ID")
    plt.xlabel("Event ID")
    plt.title(f"{data_name}: {mode}")
    plt.savefig(f'./causal_graph_plot/{data_name}_{mode}.png', bbox_inches='tight')
    plt.clf()

plot_causal_graph("amazon", "train")
plot_causal_graph("amazon", "test")

plot_causal_graph("retweet", "train")
plot_causal_graph("retweet", "test")

plot_causal_graph("taxi", "train")
plot_causal_graph("taxi", "test")

plot_causal_graph("taobao", "train")
plot_causal_graph("taobao", "test")

plot_causal_graph("so", "train")
plot_causal_graph("so", "test")