
import sys
sys.path.append("../")
sys.path.append("../../")

from tqdm import tqdm
import numpy as np
from bert_pytorch.dataset import WordVocab

import torch
import argparse
from torch import Tensor

from training_data_MaskGAE import LogGraphDataset, find_all_edges, find_all_negative_edges
from model_MaskGAE import GNNEncoder, EdgeDecoder, MaskGAE
from mask_MaskGAE import MaskEdge

from torch.utils.data import DataLoader

import random
from sklearn.metrics import roc_auc_score

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

data_path='../output/tbird/train_with_time.pt'
vocab_path='../output/tbird/vocab.pkl'
emb_path = '../output/tbird/bert/embedding.pt'
# data_iter[i]: represent a sequence of event id
with open(data_path, 'r') as f:
    data_iter = f.readlines()

# embedding table trained from BERT used as input features
feat = torch.load(emb_path)
n_feat = feat.shape[1]

vocab = WordVocab.load_vocab(vocab_path)
print("vocab Size: ", len(vocab))
vocab_dict = vocab.stoi # string-to-index
n = len(vocab) # size of embedding table

# shuffle

all_seqs = [[vocab_dict[ln] for ln in line.split()] for line in data_iter]
count = [len(seq) for seq in all_seqs]
filtered = [x for x, m in zip(all_seqs, count) if m > 10]

idx = list(range(len(filtered)))         # [0, 1, ..., n-1]
random.shuffle(idx) 
shuffled_filtered = [filtered[i] for i in idx]

# dataset = LogGraphDataset(data_iter, vocab_dict)
# dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=dataset.collate)

# count_all = len(dataloader)
train_size = 5000
train_iter = shuffled_filtered[ :train_size]
valid_iter = shuffled_filtered[train_size: ]

train_set = LogGraphDataset(train_iter, vocab_dict)
train_dataloader = DataLoader(train_set, batch_size=1, num_workers=1, collate_fn=train_set.collate)
all_negative_edges_train = find_all_not_existing_edges(train_iter, vocab_dict)

valid_set = LogGraphDataset(valid_iter, vocab_dict)
valid_dataloader = DataLoader(valid_set, batch_size=1, num_workers=1, collate_fn=valid_set.collate)
all_negative_edges_valid = find_all_not_existing_edges(valid_iter, vocab_dict)

encoder = GNNEncoder(
    in_channels=n_feat, hidden_channels=args.encoder_channels, out_channels=args.hidden_channels,
    num_layers=args.encoder_layers, dropout=args.encoder_dropout,
    bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder = EdgeDecoder(
    args.hidden_channels, args.decoder_channels,
    num_layers=args.decoder_layers, dropout=args.decoder_dropout)

mask = MaskEdge(p=args.p)

model = MaskGAE(encoder, edge_decoder, mask).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# inductive: multiple graphs
# input to the model: edge_index_unmasked; n, feat; edge_index_masked_for_loss 

# node input feature: trained embedding from BERT
x = feat.to(device).to(dtype=torch.float32)
accum_steps = 1000
# train 200 epochs
train_loss = []
valid_auc = []
for it in range(5):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    for step, batch in enumerate(train_dataloader, start=1):
        # model input (one graph: unmasked all edges; pre-trained node embeddings)
        edge_index = torch.tensor(batch).to(device).to(dtype=torch.long)
        # define training here
        loss = model(x, edge_index, all_negative_edges_train)
        loss = loss / accum_steps
        loss.backward()
        running_loss += loss.item()
        if step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(running_loss)
            print(f"Epoch: {it}; Total avg Loss (1000): {running_loss}")
            running_loss = 0.0
            #######
    model.eval()
    y_list = []; pred_list = []
    for _, batch in enumerate(valid_dataloader, start=1):
        edge_index = torch.tensor(batch).to(device).to(dtype=torch.long)
        y, pred = model.evaluate(x, edge_index, all_negative_edges_valid)
        y_list.append(y)
        pred_list.append(pred)
    ####
    all_pred = torch.cat(pred_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    all_y, all_pred = all_y.cpu().numpy(), all_pred.cpu().numpy()
    auc = roc_auc_score(all_y, all_pred)
    valid_auc.append(auc)
    print(f"Epoch: {it}; validation roc_auc: {auc}")
        
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

        

