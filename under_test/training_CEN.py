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
from TGN_model.rrgcn import RecurrentRGCNCEN
from TGN_model.tkg_utils_dgl import build_sub_graph

def edge_list_to_triples(edge_list):
    # edge_list: (2 x num_edges)
    # to: (num_edges x 3), each row: src, rel, dst; dummy rel for data without edge relation type
    dummy_rel = 0
    src = edge_list[0]
    dst = edge_list[1]
    num_edges = src.shape[0]
    rel = np.full(num_edges, dummy_rel, dtype=np.int64)
    triples = np.stack([src, rel, dst], axis=1)
    return triples


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)

args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, n = prepare_seq(
    data_name="amazon")
num_nodes = n
num_rels = 1 # dummy relation


model = RecurrentRGCNCEN(
    args.decoder,
    args.encoder,
    num_nodes,
    num_rels,
    args.n_hidden,
    args.opn,
    sequence_len=args.train_history_len,
    num_bases=args.n_bases,
    num_basis=args.n_basis,
    num_hidden_layers=args.n_layers,
    dropout=args.dropout,
    self_loop=args.self_loop,
    skip_connect=args.skip_connect,
    layer_norm=args.layer_norm,
    input_dropout=args.input_dropout,
    hidden_dropout=args.hidden_dropout,
    feat_dropout=args.feat_dropout,
    entity_prediction=args.entity_prediction,
    relation_prediction=args.relation_prediction,
    use_cuda=use_cuda,
    gpu = args.gpu)

# in training: encode the input sequence; predict the output sequence step-by-step
# at each step, compute loss against ground-truth edge; update embeddings using the true edge (teacher forcing)
# (autoregressive decoding) at inference: do not use teacher forcing, feed the predicted edges back into the history 
# the autoregressive decoding does not give comparable results if other method predict the whole graph at once
# since edge may reoccur at different time (solution: inference autoregressively then aggregate into one graph)
# we just want to compare whether autoregressively encode the input sequence is effective
def train_cen_seq2seq(model, optimizer, train_dataloader, num_nodes, num_rels, history_len, device="cuda"):
    # train_dataloader batch size should be 1
    model.train()
    total_loss = 0.0
    num_updates = 0
    for batch in train_dataloader:
        input_edges, output_edges = batch
        # build local timeline
        timeline = np.concatenate((input_edges, output_edges), axis=1)
        T_in = input_edges.shape[1]
        T_total = input_edges.shape[1] + output_edges.shape[1]
        # skip if not enough history
        if T_in < history_len:
            continue
        # --- 1. warm-up: consume input sequence ---
        for t in range(history_len, T_in):
            history = timeline[:, t - history_len: t]
            target = timeline[:, t]
            # G_{t-h}, ..., G_{t-1} ---> G_{t}
            # create G_{t-h}, ..., G_{t-1}
            history_triples = edge_list_to_triples(history)
            # each timestamp has one edge
            history_graphs = [
                build_sub_graph(num_nodes, num_rels, np.expand_dims(snap, axis=0), True, device)
                for snap in history_triples
            ]
            target_tensor = torch.tensor(
                target, dtype=torch.long, device=device
            ) # check this
            # here the node feature is not used?
            loss = model.get_loss(
                glist=history_graphs,
                triples=target_tensor,
                prev_model=None,
                use_cuda=(device=="cuda")
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # --- 2. seq2seq prediction with teacher forcing ---
        for t in range(T_in, T_total):
            history = timeline[t-history_len:t]
            target  = timeline[t]
            history_graphs = [
                build_sub_graph(num_nodes, num_rels, snap, device=="cuda", 0)
                for snap in history
            ]
            target_tensor = torch.tensor(
                target, dtype=torch.long, device=device
            )
            loss = model.get_loss(
                glist=history_graphs,
                triples=target_tensor,
                prev_model=None,
                use_cuda=(device=="cuda")
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_updates += 1
    return total_loss / max(1, num_updates)

def cen_autoregressive_decode(model, input_sequence, num_nodes, num_rels, history_len, pred_len, device="cuda"):
    model.eval()
    # ---- 1. Initialize local timeline with input ----
    timeline = list(input_sequence)  # copy
    T_in = len(input_sequence)
    assert T_in >= history_len, "Input sequence too short"
    predicted_sequence = []
    # ---- 2. Autoregressive decoding ----
    for step in range(pred_len):
        # 2.1 Construct history window (last H edges)
        history_snaps = timeline[-history_len:]
        history_graphs = [
            build_sub_graph(
                num_nodes=num_nodes,
                num_rels=num_rels,
                triples=snap,                 # [1,3]
                use_cuda=(device == "cuda"),
                gpu=0
            )
            for snap in history_snaps
        ]
        # 2.2 Forward pass: get scores for next edge
        # This internally:
        #   - evolves embeddings through history_graphs
        #   - applies ConvTransE decoder
        with torch.no_grad():
            scores = model.predict_next(
                glist=history_graphs,
                use_cuda=(device == "cuda")
            )
            # scores: [num_nodes] or [num_nodes, num_rels, num_nodes]
            # depending on implementation
        # 2.3 Decode the next edge
        # Here we assume:
        #   - relation is fixed or known (often r=0)
        #   - predict (subject, object) or (object | subject)
        u_pred, r_pred, v_pred = decode_edge_from_scores(scores)
        next_edge = np.array([[u_pred, r_pred, v_pred]])
        # 2.4 Append prediction
        timeline.append(next_edge)
        predicted_sequence.append(next_edge)
    return predicted_sequence

# generate history graph
# train_list[t] = all (subject, relation, object) triples at timestamp t
# train_list is time-ordered; index = discrete time
# train_list[t]: (num_edges_at_t, 3); all edges whose timestamp is exactly t
# does not use cumulative graphs by default
# h: history length used in prediction
# use consecutive past snapshots for prediction
# G_{t-h}, ..., G_{t-1} ---> G_{t} 

history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

loss= model.get_loss(history_glist, output[-1], None, use_cuda)
losses.append(loss.item())

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
optimizer.step()
optimizer.zero_grad()