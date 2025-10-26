import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
from .time_embed import TimeEmbedding

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, embed_size, hidden_size, max_len, dropout=0.1):
        """
        :param embed_size: total vocab size
        :param hidden_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(embed_size=embed_size, hidden_size=hidden_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.position(sequence)
        x = x + self.token(sequence)
        return self.dropout(x)
