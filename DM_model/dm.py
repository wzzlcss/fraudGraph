import torch
import torch.nn as nn
import torch.nn.functional as F

def make_gamma_schedule(T=100, gamma_start=1e-4, gamma_end=0.2):
    # T: total number of diffusion steps
    # gamma: the probability of getting replaced by pure noise during the forward process
    # the direct analog of the Gaussian noise variance beta_t in DDPM
    return torch.linspace(gamma_start, gamma_end, T)  # simple linear schedule

def q_sample_discrete(A0, t, gammas): # Forward diffusion: q(A_t | A_0, t)
    # A0: [B, N, N] ground-truth adjacency at time t (A_t)
    # t:  [B] diffusion timestep indices in [1..T]
    # gammas: [T] noise rates gamma_t
    # Returns: A_noisy [B, N, N] int64
    B, N, _ = A0.shape
    # gather gamma_t per sample
    gamma_t = gammas[t - 1].view(B, 1, 1)  # [B, 1, 1]
    # mask for which entries to randomize
    noise_mask = torch.bernoulli(gamma_t.expand(B, N, N)).bool()  # [B, N, N]
    # random classes (0 for no edges; 1 for has edges)
    random_vals = torch.randint(low=0, high=2, size=(B, N, N), device=A0.device)
    A_noisy = A0.clone()
    A_noisy[noise_mask] = random_vals[noise_mask]
    return A_noisy

class GraphDenoiser(nn.Module):
    # graph denoiser with conditional input
    # take noisy graph at diffusion step t
    # output logits over edges conditioned on (A_prev, X_prev)
    def __init__(self, d_model, T, gnnencoder):
        super().__init__()
        self.d_model = d_model
        self.T = T

        # Time-step embedding
        self.time_embed = nn.Embedding(T + 1, d_model)

        # Encoder to learn the embedding for previous graph G_{t-1} = (A_prev, X_prev)
        self.prev_encoder = gnnencoder

        # Embedding for "no condition" (classifier-free guidance)
        self.no_cond_embed = nn.Parameter(torch.zeros(d_model))

        # Embedding for noisy adjacency at current step
        # We embed discrete edge types and aggregate into node-level info
        self.edge_embed = nn.Embedding(2, d_model)

        # Simple node-level fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model + d_model, d_model),  # H_prev + H_noisy + time_emb
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        # Final edge predictor (pairwise scores for each edge type)
        self.edge_classifier = nn.Linear(2 * d_model, 2)

    def encode_prev_graph(self, edge_prev_in, feat_prev_in):
        # Encode G_{t-1}. edge_prev_in: list of edge index; feat_prev_in: [B, N, d_x]
        # Returns: H_prev: [B, N, d_model]
        H_prev = []
        B = len(edge_prev_in)
        for i in range(B):
            H_prev.append(self.prev_encoder(feat_prev_in, edge_prev_in[i]))
        ###
        H_prev = torch.stack(H_prev, dim=0)
        return H_prev

    def encode_noisy_graph(self, adj_noisy):
        # Encode noisy adjacency into node-level embeddings: adj_noisy: [B, N, N]
        # Returns: H_noisy: [B, N, d_model]
        B, N, _ = adj_noisy.shape
        # edge embeddings: [B, N, N, d_model]
        edge_emb_per_node = self.edge_embed(adj_noisy)
        # simple pooling over neighbors to get node embeddings
        # H_noisy[i] = mean_j E_emb[i,j] + E_emb[j,i]
        H_noisy = edge_emb_per_node.mean(dim=2)  # [B, N, d_model] (incoming)
        return H_noisy

    def forward(self, adj_noisy, t, edge_prev_in=None, feat_prev_in=None):
        # A_noisy: [B, N, N] noisy adjacency at step t
        # t: [B] diffusion step indices in [1..T]
        # A_prev: [B, N, N]; X_prev: [B, N, d_x] or None
        # Return logits_edges: [B, N, N, 2]

        B, N, _ = adj_noisy.shape
        # time embedding
        t_emb = self.time_embed(t)  # [B, d_model]
        t_emb = t_emb.view(B, 1, self.d_model).expand(B, N, self.d_model)  # [B, N, d_model]

        # encode noisy graph at time t
        H_noisy = self.encode_noisy_graph(adj_noisy)  # [B, N, d_model]

        # encode condition graph or use "no condition"
        if (edge_prev_in is None) and (feat_prev_in is None):
            H_prev = self.no_cond_embed.view(1, 1, self.d_model).expand(B, N, self.d_model)
        else:
            H_prev = self.encode_prev_graph(edge_prev_in, feat_prev_in)  # [B, N, d_model]

        # fuse node representations
        H_fused = self.fusion_mlp(
            torch.cat([H_noisy, H_prev, t_emb], dim=-1))  # [B, N, d_model]

        # build pairwise edge logits from node embeddings
        # simple bilinear-like: concat h_i and h_j, then classify
        H_i = H_fused.unsqueeze(2).expand(B, N, N, self.d_model)  # [B, N, N, d_model]
        H_j = H_fused.unsqueeze(1).expand(B, N, N, self.d_model)  # [B, N, N, d_model]

        H_pair = torch.cat([H_i, H_j], dim=-1)  # [B, N, N, 2*d_model]

        logits_edges = self.edge_classifier(H_pair)  # [B, N, N, 2]
        return logits_edges

