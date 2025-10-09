
import sys
sys.path.append("../")
sys.path.append("../../")

from tqdm import tqdm
import numpy as np
from bert_pytorch.dataset import WordVocab

import torch
import argparse
from torch import Tensor

from training_data_MaskGAE import LogGraphDatasetAdjPair, find_all_negative_edges, all_edges
from model_MaskGAE import GNNEncoder, EdgeDecoder, CondGAE
from mask_MaskGAE import MaskEdge

from torch.utils.data import DataLoader

import random
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import add_self_loops

parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
# parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2025, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.3, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.3, help='Dropout probability of decoder. (default: 0.3)')
# parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
# parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')
parser.add_argument('--batch_size', type=int, default=6000, help='Number of training batch size.')

# parser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 300)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-LinkPred.pt", help="save path for model. (default: MaskGAE-LinkPred.pt)")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

# window_size=128 #trainer.window_size
# adaptive_window=True #trainer.adaptive_window
# valid_size=0.1 #trainer.valid_ratio
# sample_ratio=1 #trainer.sample_ratio
# scale=None #trainer.scale
# scale_path='../output/tbird/bert/scale.pkl' #trainer.scale_path
# seq_len=512 #trainer.seq_len
# min_len=10 #trainer.min_len

# ########
# data_path='../output/tbird/train'
# with open(data_path, 'r') as f:
#     data_iter = f.readlines()

# data_path='../output/tbird/test_abnormal'
# with open(data_path, 'r') as f:
#     test_iter = f.readlines()

# vocab_path='../output/tbird/vocab.pkl'
# vocab = WordVocab.load_vocab(vocab_path)
# print("vocab Size: ", len(vocab))
# vocab_dict = vocab.stoi # string-to-index
# n = len(vocab) # size of embedding table

# all_unique_edges_train = find_all_edges(data_iter, vocab_dict)
# all_unique_edges_test = find_all_edges(test_iter, vocab_dict)
# ########

def find_all_not_existing_edges(data_iter, vocab_dict):
    all_unique_edges = find_all_edges(data_iter, vocab_dict)
    all_negative_edges = find_all_negative_edges(all_unique_edges, n, exclude_diag=True)
    all_negative_edges = torch.tensor(all_negative_edges).to(device).to(dtype=torch.long)
    return all_negative_edges

train_data_path='../output/tbird/train_with_time'
valid_data_path='../output/tbird/valid_with_time'
vocab_path='../output/tbird/vocab.pkl'
emb_path = '../output/tbird/bert/embedding.pt'
# data_iter[i]: represent a sequence of event id
with open(train_data_path, 'r') as f:
    train_iter = f.readlines()

with open(valid_data_path, 'r') as f:
    valid_iter = f.readlines()

with open(data_path, 'r') as f:
    data_iter = f.readlines()

# embedding table trained from BERT used as input features
feat = torch.load(emb_path)
n_feat = feat.shape[1]

vocab = WordVocab.load_vocab(vocab_path)
print("vocab Size: ", len(vocab))
vocab_dict = vocab.stoi # string-to-index
n = len(vocab) # size of embedding table

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
    return eventSeq_sorted_pair

train_data = filter_raw_seq(train_iter, vocab_dict, min_seq_len = 10)  
valid_data = filter_raw_seq(valid_iter, vocab_dict, min_seq_len = 10)        

# eventSeq = [[vocab_dict[ln] for ln in (line.split()[2]).split(",")] for line in data_iter]
# timeSeq = [[int(ln) for ln in (line.split()[0]).split(",")] for line in data_iter]
# count = [len(seq) for seq in eventSeq]
# filtered_eventSeq = [x for x, m in zip(eventSeq, count) if m > 10]
# filtered_timeSeq = [x for x, m in zip(timeSeq, count) if m > 10]
# start_time = [ln[0] for ln in filtered_timeSeq]
# # reorder Seq by the starting time stamp
# idx = sorted(range(len(start_time)), key=start_time.__getitem__)
# eventSeq_sorted_pair = [(filtered_eventSeq[idx[i]], filtered_eventSeq[idx[i+1]]) for i in range(len(idx)-1)]
# timeSeq_sorted_pair = [(filtered_timeSeq[idx[i]], filtered_timeSeq[idx[i+1]]) for i in range(len(idx)-1)]

# idx = list(range(len(filtered)))         # [0, 1, ..., n-1]
# random.shuffle(idx) 
# shuffled_filtered = [filtered[i] for i in idx]

# dataset = LogGraphDataset(data_iter, vocab_dict)
# dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=dataset.collate)

# count_all = len(dataloader)
# train_size = 32000
# train_iter = eventSeq_sorted_pair[ :train_size]
# valid_iter = eventSeq_sorted_pair[train_size: ]

train_set = LogGraphDatasetAdjPair(train_data)
train_dataloader = DataLoader(train_set, batch_size=1, num_workers=1, collate_fn=train_set.collate)
# all_negative_edges_train = find_all_not_existing_edges(train_iter, vocab_dict)

valid_set = LogGraphDatasetAdjPair(valid_data)
valid_dataloader = DataLoader(valid_set, batch_size=1, num_workers=1, collate_fn=valid_set.collate)
# all_negative_edges_valid = find_all_not_existing_edges(valid_iter, vocab_dict)

encoder = GNNEncoder(
    in_channels=n_feat, hidden_channels=args.encoder_channels, out_channels=args.hidden_channels,
    num_layers=args.encoder_layers, dropout=args.encoder_dropout,
    bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder = EdgeDecoder(
    args.hidden_channels, args.decoder_channels,
    num_layers=args.decoder_layers, dropout=args.decoder_dropout)

# mask = MaskEdge(p=args.p)

model = CondGAE(encoder, edge_decoder).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def compute_batch_auc(y_list, pred_list):
    all_pred = torch.cat(pred_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    all_y, all_pred = all_y.cpu().numpy(), all_pred.cpu().numpy()
    auc = roc_auc_score(all_y, all_pred)
    return auc

# inductive: multiple graphs
# input to the model: edge_index_unmasked; n, feat; edge_index_masked_for_loss 
n_train_sample = len(train_dataloader)

all_edges_n = all_edges(n)
all_edges_n = torch.tensor(all_edges_n).to(device).to(dtype=torch.long)

edge_index_empty = torch.empty(2, 0, dtype=torch.long)
self_loop_only, _ = add_self_loops(edge_index_empty, num_nodes = n)
self_loop_only = self_loop_only.to(device).to(dtype=torch.long)

# node input feature: trained embedding from BERT
x = feat.to(device).to(dtype=torch.float32)

accum_steps = 1000
remainder = n_train_sample % accum_steps
last_chunk_size = remainder if remainder != 0 else acc_steps
last_chunk_step = n_train_sample - remainder

train_loss = []
valid_auc_cond = []
valid_auc_bert = []
valid_auc_rand_input = []
best_auc = 0.0
best_model = None

# one epoch
model.train()
optimizer.zero_grad()
running_loss = 0.0
for step, batch in enumerate(train_dataloader, start=1):
    # model input (one graph: unmasked all edges; pre-trained node embeddings)
    input_edge, output_edge = batch
    negative_edges_in_output = find_all_negative_edges(output_edge, n, exclude_diag=True)
    negative_edges_in_output = torch.tensor(negative_edges_in_output).to(device).to(dtype=torch.long)
    input_edge = torch.tensor(input_edge).to(device).to(dtype=torch.long)
    output_edge = torch.tensor(output_edge).to(device).to(dtype=torch.long)
    # define training norm
    is_acc_step = ((step + 1) % accum_steps == 0)
    is_last_chunk_step = ((step + 1) >= last_chunk_step)
    demon = accum_steps if not is_last_chunk_step else remainder
    is_last_step = ((n + 1) == n_train_sample)
    # define training step
    loss = model(x, input_edge, output_edge, negative_edges_in_output)
    loss = loss / demon
    loss.backward()
    running_loss += loss.item()
    if is_acc_step or is_last_step: # make change here
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(running_loss)
        print(f"Total avg Loss ({step+1} / {n_train_sample}): {running_loss}")
        running_loss = 0.0
        ####### validation #######
        model.eval()
        y_list_cond = []; pred_list_cond = []
        y_list_rand_input = []; pred_list_rand_input = []
        y_list_bert = []; pred_list_bert = []
        # y_list_rand = []; pred_list_rand = []
        for _, batch in enumerate(valid_dataloader, start=1):
            input_edge, output_edge = batch
            negative_edges_in_output = find_all_negative_edges(output_edge, n, exclude_diag=True)
            negative_edges_in_output = torch.tensor(negative_edges_in_output).to(device).to(dtype=torch.long)
            input_edge = torch.tensor(input_edge).to(device).to(dtype=torch.long)
            output_edge = torch.tensor(output_edge).to(device).to(dtype=torch.long)
            # sample negative edges
            num_negative_samples = output_edge.shape[1]
            neg_samples_idx = torch.randint(
                negative_edges_in_output.shape[1], (num_negative_samples,), device=negative_edges_in_output.device)
            neg_samples = negative_edges_in_output[:, neg_samples_idx]
            #
            y_cond, pred_cond = model.evaluate(x, input_edge, output_edge, neg_samples)
            y_list_cond.append(y_cond); pred_list_cond.append(pred_cond.squeeze())
            # input a random graph (with the same number of edges as the input) as a condition
            num_input_edge = input_edge.shape[1]
            all_edges_n_samples_idx = torch.randint(
                all_edges_n.shape[1], (num_input_edge,), device=all_edges_n.device)
            random_input_edge = all_edges_n[:, all_edges_n_samples_idx]
            y_rand_input, pred_rand_input = model.evaluate(x, random_input_edge, output_edge, neg_samples)
            y_list_rand_input.append(y_rand_input); pred_list_rand_input.append(pred_rand_input.squeeze())
            # decoder only
            y_bert, pred_bert = model.evaluate(x, self_loop_only, output_edge, neg_samples)
            y_list_bert.append(y_bert); pred_list_bert.append(pred_bert.squeeze())
            # input a random graph; predit a random graph
            # num_output_edge = output_edge.shape[1]
            # all_edges_n_samples_idx_output = torch.randint(
            #     all_edges_n.shape[1], (num_output_edge,), device=all_edges_n.device)
            # random_output_edge = all_edges_n[:, all_edges_n_samples_idx_output]
            # y_rand, pred_rand = model.evaluate(x, random_input_edge, random_output_edge, neg_samples)
            # y_list_rand.append(y_rand); pred_list_rand.append(pred_rand.squeeze())
        ####
        auc_cond = compute_batch_auc(y_list_cond, pred_list_cond)
        auc_rand_cond = compute_batch_auc(y_list_rand_input, pred_list_rand_input)
        auc_bert = compute_batch_auc(y_list_bert, pred_list_bert)
        # auc_rand = compute_batch_auc(y_list_rand, pred_list_rand)
        valid_auc_cond.append(auc_cond)
        # valid_auc_rand.append(auc_rand)
        valid_auc_bert.append(auc_bert)
        valid_auc_rand_input.append(auc_rand_cond)
        if auc_cond > best_auc:
            best_auc = auc_cond
            torch.save({
                'args': args,
                'model_state_dict': model.state_dict(),
                'best_valid_auc': best_auc
            }, 'best_CondGAE.pt')
        print(f"Step: {step+1}; validation roc_auc (true condition): {auc_cond}; (random input): {auc_rand_cond}; (bert): {auc_bert}")
        model.train()

# test valid: input edge is empty graph

# test valid: input edge is a random
        
import matplotlib.pyplot as plt

train_idx = [i for i in range(len(train_loss))]
valid_idx = [i for i in range(len(train_loss)) if (i+1) % 5 == 0 ]

plt.clf()  
plt.plot(train_idx, train_loss)
plt.ylabel("Training loss")
plt.xlabel("number of batches")
plt.savefig(f'./training_loss', bbox_inches='tight')
plt.clf()

plt.clf()  
plt.plot(valid_idx, valid_auc)
plt.ylabel("Validation AUC")
plt.xlabel("number of batches")
plt.xticks(valid_idx) 
plt.savefig(f'./valid_auc', bbox_inches='tight')
plt.clf()

        

