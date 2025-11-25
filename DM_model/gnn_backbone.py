import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from DM_model.nn import normalization, zero_module, timestep_embedding

class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return pos_x

class GNNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
        """
        super(GNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.D = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

    def forward(self, h, e_noisy, e_cond, graph, mode="residual"):
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_noisy_in = e_noisy
        e_cond_in = e_cond

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H

        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating 
        Ah = self.A(h)  # B x V x H, source
        Bh = self.B(h)  # B x V x H, target
        Ce_noisy = self.C(e_noisy)  # B x V x V x H / E x H
        De_cond = self.D(e_cond)  # B x V x V x H / E x H (conditional graph)

        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce_noisy + De_cond  # B x V x V x H
        gates = torch.sigmoid(e)

        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates)  # B x V x H

        h = self.norm_h(
            h.view(batch_size * num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h

        e = self.norm_e(
            e.view(batch_size * num_nodes * num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        if mode == "residual":
            h = h_in + h
            e = e_in + e

        return h, e

    def aggregate(self, Vh, graph, gates):
        Vh = gates * Vh  # B x V x V x H
        if (self.aggregation) == "mean":
            return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
        elif (self.aggregation) == "max":
            return torch.max(Vh, dim=2)[0]
        else:
            return torch.sum(Vh, dim=2)

class GNNEncoder(nn.Module):
    def __init__(
            self, n_layers, in_channels, hidden_dim, out_channels=2, aggregation="sum", norm="layer",
            learn_norm=True, track_norm=False, gated=True,):
        super(GNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        time_embed_dim = hidden_dim // 2
        self.node_embed = nn.Linear(in_channels, hidden_dim) # transform the input node embedding
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

        self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        )
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
            for _ in range(n_layers)
        ])
        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(time_embed_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])
        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim) # set the initial parameters to zero
            ),
            ) for _ in range(n_layers)
        ])

    def forward(self, x, graph_noisy, graph_cond, timesteps):
        """
        Args:
            x: Input node coordinates (B x V x H_init)
            graph_noisy: Noisy graph adjacency matrices at current diffusion step (B x V x V) # transformed
            graph_cond: conditional graph adjacency (B x V x V)
            timesteps: Input node timesteps (B)
        Returns:
            Updated edge features (B x V x V)
        """
        x = self.node_embed(x)

        # embed adj: (B x V x V) --> (B x V x V x H)
        e_noisy = self.edge_embed(self.edge_pos_embed(graph_noisy))
        e_cond = self.edge_embed(self.edge_pos_embed(graph_cond))
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))

        # in GNN aggregation: no graph is used; graph is only used in edge embedding
        graph = torch.ones_like(graph_noisy).long()

        for gnn_layer, time_layer, out_layer in zip(self.gnn_layers, self.time_embed_layers, self.per_layer_out):
            x_in = x; e_noisy_in = e_noisy; e_cond_in = e_cond
            # update node embedding & edge embedding using both e_noisy and e_cond
            x, e_noisy = gnn_layer(x, e_noisy, e_cond, graph, mode="direct")
            e_noisy = e_noisy + time_layer(time_emb)[:, None, None, :]
            x = x_in + x
            e_noisy = e_noisy_in + out_layer(e_noisy)
        
        e = self.out(e_noisy.permute((0, 3, 1, 2)))
        return e
