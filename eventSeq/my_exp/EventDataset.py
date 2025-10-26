from torch.utils.data import Dataset
import torch
import random
import numpy as np
from collections import defaultdict

class EventDataset(Dataset):
    def __init__(self, event_corpus, vocab_size, seq_len, predict_mode=False, mask_ratio=0.15):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.predict_mode = predict_mode
        self.event_corpus = event_corpus
        self.mask_ratio = mask_ratio  
        self.corpus_lines = len(event_corpus)

        # self.pad_index = 0 + vocab_size
        # self.unk_index = 1 + vocab_size # for unknown id
        # self.eos_index = 2 + vocab_size
        # self.sos_index = 3 + vocab_size
        # self.mask_index = 4 + vocab_size      
        self.pad_index = 0 + vocab_size
        self.mask_index = 1 + vocab_size # valid event id starts from 0

    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, idx):
        seq = self.event_corpus[idx]
        # k_label: 0 if not a masked token; 
        # if not 0, the corresponding token has been masked and this is the original value
        seq_masked, seq_label = self.random_item(seq)
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # k = [self.sos_index] + seq_masked
        # k_label = [self.pad_index] + seq_label
        # return k, k_label
        return seq_masked, seq_label

    def random_item(self, seq):
        tokens = list(seq.numpy())
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                if self.predict_mode:
                    tokens[i] = self.mask_index
                    output_label.append(token)
                    continue

                prob /= self.mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab_size)

                # 10% randomly change token to current token
                else:
                    tokens[i] = token

                output_label.append(token)

            else:
                tokens[i] = token
                output_label.append(self.pad_index)

        return tokens, output_label


    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]

        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = defaultdict(list)
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]

            padding = [self.pad_index for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding)

            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)

        output["bert_input"] = torch.tensor(output["bert_input"], dtype=torch.long)
        output["bert_label"] = torch.tensor(output["bert_label"], dtype=torch.long)

        return output
