import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from DM_model.gnn_backbone import GNNEncoder
from DM_model.diffusion_schedulers import CategoricalDiffusion, InferenceSchedule

class DMModel(nn.Module):
    def __init__(self, args, device):
        super(DMModel, self).__init__()
        # catgorical diffusion
        self.args = args
        self.device = device
        self.diffusion_steps = self.args.diffusion_steps
        self.model = GNNEncoder(
            n_layers=self.args.n_layers,
            in_channels=self.args.in_channels,
            hidden_dim=self.args.hidden_dim,
            out_channels=2,
            aggregation=self.args.aggregation,
            # use_activation_checkpoint=self.args.use_activation_checkpoint,
        ).to(self.device)
        self.diffusion_schedule = self.args.diffusion_schedule # 'linear' or 'cosine'
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule)

    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step.
        See https://arxiv.org/pdf/2107.03006.pdf for details.
        """
        diffusion = self.diffusion

        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        return xt
    
    def forward(self, x, adj_noisy, adj_cond, t):
        return self.model(x, adj_noisy, adj_cond, t)

    def categorical_training_step(self, batch, feat_batch):
        adj_in, adj_out = batch # adj_out: target clean adj; adj_in: conditional adj
        batch_size = adj_in.shape[0]
        # sample diffusion steps
        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        # Sample noisy xt from diffusion: adj_out is the x0
        adj_matrix_onehot = F.one_hot(adj_out.long(), num_classes=2).float()

        xt = self.diffusion.sample(adj_matrix_onehot, t)
        xt = xt * 2 - 1 # {0, 1} -> {-1, 1}
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt)) # add small noise

        t = torch.from_numpy(t).float().view(adj_out.shape[0])

        adj_cond = adj_in

        # Denoise
        x0_pred = self.forward(
            feat_batch.float().to(self.device),
            xt.float().to(self.device),
            adj_cond.float().to(self.device),
            t.float().to(self.device),
        )
        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, adj_out.long().to(self.device))
        # self.log("train/loss", loss)
        return loss

    def categorical_denoise_step(self, t1, t2, feat_batch, xt, adj_cond):
        with torch.no_grad():
            t1 = torch.from_numpy(t1).view(1)
            x0_pred = self.forward(
                feat_batch.float().to(self.device),
                xt.float().to(self.device),
                adj_cond.float().to(self.device),
                t1.float().to(self.device),
            )
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            xt = self.categorical_posterior(
                target_t=t2, t=t1, x0_pred_prob=x0_pred_prob.cpu(), xt=xt)
        return xt

    def test_set(self, batch, feat_batch):
        adj_in, adj_out = batch # adj_out: target clean adj; adj_in: conditional adj
        # start from a pure random graph
        xt = torch.randn_like(adj_out.float())
        xt = (xt > 0).long() # this maps to {0, 1} but in training, eges are in {-1, 1} + noise
        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(
            inference_schedule=self.args.inference_schedule,
            T=self.diffusion.T, inference_T=steps)
        # diffusion iteration
        for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            adj_cond = adj_in
            # update
            xt = self.categorical_denoise_step(t1, t2, feat_batch, xt, adj_cond) # change and implement this

        adj_mat = xt.float().cpu().detach().numpy()
        return adj_mat


