import pickle
import torch
import numpy as np
import argparse
import gc
import random
import os
from EventDataset import EventDataset
from torch.utils.data import DataLoader

import sys
sys.path.append("../")
from bert_pytorch import BERT, BERTLog
from bert_pytorch.trainer import BERTTrainer, ScheduledOptim

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default="amazon", help='event sequence dataset name')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='masking ratio in training BERT')
parser.add_argument('--seq_len', type=int, default=90, help='sequence length in traing BERT (for determining positional embedding)')
parser.add_argument('--batch_size', type=int, default=20, help='batch size in training BERT')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers in datalowader')
# model
parser.add_argument('--max_len', type=int, default=90, help='seq max length for positional embedding')
parser.add_argument('--hidden', type=int, default=64, help='hidden size')
parser.add_argument('--layers', type=int, default=2, help='number of transformer layers')
parser.add_argument('--attn_heads', type=int, default=4, help='number of attention heads')
parser.add_argument("--device", type=int, default=0)
# training
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.00)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--n_epochs_stop', type=int, default=10)
parser.add_argument('--exp_name', type=str, default="train_bert_embedding")
args = parser.parse_args()

def load_event(data_name, mode="train"):
    dataset_dir = f"../data/{data_name}/{mode}.pkl"
    with open(dataset_dir, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[mode]
    # time_seq = [[x["time_since_start"] for x in seq] for seq in data]
    # time_seq = [torch.tensor(seq[1:]) for seq in time_seq]
    event_seq = [[x["type_event"] for x in seq] for seq in data]
    event_seq = [torch.tensor(seq[1:]) for seq in event_seq]
    seq_len_all = np.array([len(seq) for seq in event_seq])
    print(
        "Loading", data_name, ":", mode,
        "\n Total number of event sequences: ", len(seq_len_all), \
        "\n Length of event sequences: mean", seq_len_all.mean(), \
        "\n median", np.median(np.median(seq_len_all)), \
        "\n min", np.min(seq_len_all), \
        "\n max", np.max(seq_len_all))
    return event_seq, num_types

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def start_iteration(surfix_log, trainer):
    print("Training Start")
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        print("\n")
        _ = trainer.train(epoch)
        avg_loss = trainer.valid(epoch)
        trainer.save_log(bert_dir, surfix_log)
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save(bert_dir)
            epochs_no_improve = 0
            epochs_no_improve += 1
        #############
        if epochs_no_improve == n_epochs_stop:
            print("Early stopping")
            break

def plot_train_valid_loss(surfix_log):
    train_loss = pd.read_csv(bert_dir + f"train{surfix_log}.csv")
    valid_loss = pd.read_csv(bert_dir + f"valid{surfix_log}.csv")
    sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
    sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
    plt.title("epoch vs train loss vs valid loss")
    plt.legend()
    plt.savefig(bert_dir + "train_valid_loss.png")
    plt.show()
    print("plot done")

def save_parameters(args, filename):
    args_dict = vars(args)
    with open(filename, "w+") as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")

##################
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
data_name = args.data_name
mask_ratio = args.mask_ratio
seq_len = args.seq_len
batch_size = args.batch_size
num_workers = args.num_workers
max_len = args.max_len
hidden = args.hidden
layers = args.layers
attn_heads = args.attn_heads
lr = args.lr
adam_beta1 = args.adam_beta1
adam_beta2 = args.adam_beta2
weight_decay = args.weight_decay
warmup_steps = args.warmup_steps
with_cuda = True
epochs = args.epochs
n_epochs_stop = args.n_epochs_stop
bert_dir = f"./{args.exp_name}/{data_name}/"

seed_everything(seed=1234)
save_parameters(args, bert_dir + "parameters.txt")


# prepare event sequence data for BERT training
# the sequence is already presented by event id

train_iter, vocab_size = load_event(data_name, mode="train")
valid_iter, vocab_size_2 = load_event(data_name, mode="dev")
assert vocab_size_2==vocab_size, "vocab size in training and validation does not match"

train_dataset = EventDataset(
    event_corpus = train_iter, vocab_size = vocab_size, seq_len = seq_len, 
    predict_mode = False, mask_ratio = mask_ratio)

valid_dataset = EventDataset(
    event_corpus = valid_iter, vocab_size = vocab_size, seq_len = seq_len, 
    predict_mode = False, mask_ratio = mask_ratio)

# the index for padding (depends on current dataset's vocab size)
pad_index = train_dataset.pad_index

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers,
    collate_fn=train_dataset.collate_fn, drop_last=True)

valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, num_workers=num_workers,
    collate_fn=train_dataset.collate_fn, drop_last=True)

del train_dataset
del valid_dataset
del train_iter
del valid_iter
gc.collect()

# embedding table size: original vocab size + one for mask index
embed_size = vocab_size + 2

print("Building BERT model")
bert = BERT(
    embed_size, max_len=max_len, hidden=hidden, n_layers=layers, attn_heads=attn_heads)

print("Creating BERT Trainer")
trainer = BERTTrainer(
    bert, embed_size=embed_size, pad_index=pad_index,
    train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
    lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay, warmup_steps=10000,
    with_cuda=with_cuda)

start_iteration("bert", trainer)

plot_train_valid_loss("bert")