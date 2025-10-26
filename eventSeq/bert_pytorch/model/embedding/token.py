import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, embed_size, hidden_size=512):
        super().__init__(embed_size, hidden_size, padding_idx=0)
