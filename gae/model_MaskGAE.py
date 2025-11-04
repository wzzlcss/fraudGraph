import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)

def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        bn=False,
        layer="gcn",
        activation="elu"
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels*heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, x, edge_index):
        # edge_index = to_sparse_tensor(edge_index, x.size(0))

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat"):

        self.eval()
        assert mode in {"cat", "last"}, mode

        edge_index = to_sparse_tensor(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        return embedding


class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    return pos_loss + neg_loss

class MaskGAE(nn.Module):
    def __init__(
        self,
        encoder,
        edge_decoder,
        mask,
        loss="ce",
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = edge_decoder
        self.mask = mask
        self.loss_fn = ce_loss

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()

    def forward(self, x, edge_index, all_negative_edges):
        remaining_edges, masked_edges = self.mask(edge_index)
        # sample same number of negative edges as the number of positive edges
        num_negative_samples = masked_edges.shape[1]
        neg_samples_idx = torch.randint(
            all_negative_edges.shape[1], (num_negative_samples,), device=all_negative_edges.device)
        neg_samples = all_negative_edges[:, neg_samples_idx]
        # forward
        z = self.encoder(x, remaining_edges)
        pos_out = self.decoder(z, masked_edges, sigmoid=True)
        neg_out = self.decoder(z, neg_samples, sigmoid=True)
        # compute loss
        return self.loss_fn(pos_out, neg_out)
    
    @torch.no_grad()
    def evaluate(self, x, edge_index, all_negative_edges):
        remaining_edges, masked_edges = self.mask(edge_index)
        num_negative_samples = masked_edges.shape[1]
        neg_samples_idx = torch.randint(
            all_negative_edges.shape[1], (num_negative_samples,), device=all_negative_edges.device)
        neg_samples = all_negative_edges[:, neg_samples_idx]
        z = self.encoder(x, remaining_edges)
        pos_out = self.decoder(z, masked_edges, sigmoid=True)
        neg_out = self.decoder(z, neg_samples, sigmoid=True)
        pred = torch.cat([pos_out, neg_out], dim=0)
        pos_y = pos_out.new_ones(pos_out.size(0))
        neg_y = neg_out.new_zeros(neg_out.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        # y, pred = y.cpu().numpy(), pred.cpu().numpy()
        return y, pred



class CondGAE(nn.Module):
    def __init__(
        self,
        encoder,
        edge_decoder,
        loss="ce",
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = edge_decoder
        self.loss_fn = ce_loss

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()

    def forward(self, x, input_edge, output_edge, neg_samples):
        # sample same number of negative edges as the number of positive edges
        # forward
        z = self.encoder(x, input_edge)
        pos_out = self.decoder(z, output_edge, sigmoid=True)
        neg_out = self.decoder(z, neg_samples, sigmoid=True)
        # compute loss
        return self.loss_fn(pos_out, neg_out)
    
    @torch.no_grad()
    def evaluate(self, x, input_edge, output_edge_pos, output_edge_neg, decoder_only=False):
        z = self.encoder(x, input_edge)
        pos_out = self.decoder(z, output_edge_pos, sigmoid=True)
        neg_out = self.decoder(z, output_edge_neg, sigmoid=True)
        pred = torch.cat([pos_out, neg_out], dim=0)
        pos_y = pos_out.new_ones(pos_out.size(0))
        neg_y = neg_out.new_zeros(neg_out.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        # y, pred = y.cpu().numpy(), pred.cpu().numpy()
        return y, pred
    
    @torch.no_grad()
    def evaluate_certain(self, z, evaluate_edge):
        out = self.decoder(z, evaluate_edge, sigmoid=True)
        preds = (out > 0.5).int()
        return preds

