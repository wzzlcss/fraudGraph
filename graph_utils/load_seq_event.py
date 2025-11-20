import pickle
import torch
import numpy as np
from .training_data_MaskGAE import LogGraphDatasetAdjPair
from .training_data_MaskGAE import AdjPairLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def load_event(data_name, mode="train"):
    dataset_dir = f"eventSeq/data/{data_name}/{mode}.pkl"
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

def event_iter_to_seq_pair(data_iter):
    eventSeq_pair = []
    eventSeq = []
    for seq in data_iter:
        total_len = len(seq)
        input_seq = list((seq[:int(total_len/2)]).numpy())
        output_seq = list((seq[int(total_len/2):]).numpy())
        eventSeq_pair.append((input_seq, output_seq))
        eventSeq.append(list(seq.numpy()))
    return eventSeq_pair, eventSeq

def prepare_seq(data_name="amazon"):
    train_iter, vocab_size = load_event(data_name, mode="train")
    test_iter, test_size_2 = load_event(data_name, mode="test")
    assert test_size_2==vocab_size, "vocab size in training and validation does not match"
    emb_path = f'eventSeq/my_exp/train_bert_embedding/{data_name}/exp3/embedding.pt'
    feat = torch.load(emb_path)
    n = feat.shape[0] # size of embedding table
    train_data, train_eventSeq = event_iter_to_seq_pair(train_iter)
    train_set = LogGraphDatasetAdjPair(train_data)
    train_dataloader = DataLoader(train_set, batch_size=1, num_workers=1, collate_fn=train_set.collate)
    valid_data, valid_eventSeq = event_iter_to_seq_pair(test_iter)
    valid_set = LogGraphDatasetAdjPair(valid_data)
    valid_dataloader = DataLoader(valid_set, batch_size=1, num_workers=1, collate_fn=valid_set.collate)
    return train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, n

def prepare_seq_adj_batch(batch_size, data_name="amazon"):
    train_iter, vocab_size = load_event(data_name, mode="train")
    test_iter, test_size_2 = load_event(data_name, mode="test")
    assert test_size_2==vocab_size, "vocab size in training and validation does not match"
    emb_path = f'eventSeq/my_exp/train_bert_embedding/{data_name}/exp3/embedding.pt'
    feat = torch.load(emb_path)
    n = feat.shape[0] # size of embedding table
    train_data, train_eventSeq = event_iter_to_seq_pair(train_iter)
    train_set = AdjPairLoader(train_data, n)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=1, collate_fn=train_set.collate)
    valid_data, valid_eventSeq = event_iter_to_seq_pair(test_iter)
    valid_set = AdjPairLoader(valid_data, n)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=1, collate_fn=valid_set.collate)
    return train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, n
