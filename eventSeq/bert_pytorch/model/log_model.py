import torch.nn as nn
import torch
from .bert import BERT

class BERTLog(nn.Module):
    """
    BERT Log Model
    """

    def __init__(self, bert: BERT, embed_size):
        """
        :param bert: BERT model which should be trained
        :param embed_size: num of rows in the embedding table for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLogModel(self.bert.hidden, embed_size)
        self.result = {"logkey_output": None}

    def forward(self, x, pad_index):
        x = self.bert(x, pad_index)

        self.result["logkey_output"] = self.mask_lm(x)

        return self.result

class MaskedLogModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, embed_size):
        """
        :param hidden: output size of BERT model
        :param embed_size: num of rows in the embedding table for masked_lm
        """
        super().__init__()
        self.linear = nn.Linear(hidden, embed_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
