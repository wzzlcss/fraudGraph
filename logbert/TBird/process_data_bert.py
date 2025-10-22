import sys
sys.path.append("../")
sys.path.append("../../")

import numpy as np

from torch.utils.data import DataLoader
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTTrainer
from bert_pytorch.dataset import LogDataset, WordVocab
from bert_pytorch.dataset.sample import generate_train_valid
from bert_pytorch.dataset.utils import save_parameters

from sklearn.model_selection import train_test_split

vocab_path = "../output/tbird/vocab.pkl"
vocab = WordVocab.load_vocab(vocab_path)
data_path = "../output/tbird/train"

def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [ln.split(",") for ln in line.split()]
    # filter the line/session shorter than 10
    if len(line) < min_len:
        return [], []
    ####
    # max seq len
    if seq_len is not None:
        line = line[:seq_len]
    ####
    if adaptive_window:
        window_size = len(line)
    ####
    line = np.array(line)
    line = line.squeeze()
    # time duration doesn't exist, then create a zero array for time
    tim = np.zeros(line.shape)
    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])
    return logkey_seqs, time_seq

#def generate_train_valid(data_path):

with open(data_path, 'r') as f:
    data_iter = f.readlines()

valid_size = 0.1
test_size = int(len(data_iter) * valid_size)

print("before filtering short session")
print("train size ", int(len(data_iter) - test_size))
print("valid size ", int(test_size))
print("="*40)

logkey_seq_pairs = []
time_seq_pairs = []
min_len = 10
window_size = 128 
seq_len = 512 # max seq length
adaptive_window = True # determine window size based on seq_len
corpus_lines = None
on_memory = True
mask_ratio = 0.5

for line in data_iter:
    # each logkeys list can have different length
    logkeys, times = fixed_window(
        line, window_size, adaptive_window, seq_len, min_len)
    logkey_seq_pairs += logkeys
    time_seq_pairs += times

logkey_seq_pairs = np.array(logkey_seq_pairs)
time_seq_pairs = np.array(time_seq_pairs)

logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(
    logkey_seq_pairs,
    time_seq_pairs,
    test_size=test_size,
    random_state=1234)

# sort seq_pairs by seq len
train_len = list(map(len, logkey_trainset))
valid_len = list(map(len, logkey_validset))

train_sort_index = np.argsort(-1 * np.array(train_len))
valid_sort_index = np.argsort(-1 * np.array(valid_len))

logkey_trainset = logkey_trainset[train_sort_index]
logkey_validset = logkey_validset[valid_sort_index]

time_trainset = time_trainset[train_sort_index]
time_validset = time_validset[valid_sort_index]

print("="*40)
print("Num of train seqs", len(logkey_trainset))
print("Num of valid seqs", len(logkey_validset))
print("="*40)

logkey_train, logkey_valid, time_train, time_valid = logkey_trainset, logkey_validset, time_trainset, time_validset

train_dataset = LogDataset(
    logkey_train, time_train, vocab, seq_len=seq_len,
    corpus_lines=corpus_lines, on_memory=on_memory, mask_ratio=mask_ratio)