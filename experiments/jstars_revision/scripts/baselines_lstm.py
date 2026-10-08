"""LSTM SST baseline in the style of Zhang et al. (2017, IEEE GRSL): each pixel's anomaly history is a univariate
sequence fed to a shared stacked LSTM, and a linear head maps the final hidden state to the 30 forecast days
(direct multi-output, as for every other model in the study). Series are instance-normalized (RevIN-style mean/std,
as in the PatchTST configuration) and de-normalized at the output. Interface matches the official DLinear/PatchTST
models: input [batch, L, pixels] -> output [batch, H, pixels].
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.lstm = nn.LSTM(1, configs.d_model, num_layers=configs.e_layers, batch_first=True,
                            dropout=configs.dropout if configs.e_layers > 1 else 0.0)
        self.head = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x):                                   # x: [B, L, P]
        b, length, p = x.shape
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        z = ((x - mean) / std).permute(0, 2, 1).reshape(b * p, length, 1)
        # Pixel sequences are processed in chunks, with activation checkpointing during training, so that memory
        # fits an 8 GB GPU; the computation is identical to a single pass.
        run = lambda c: self.lstm(c)[0][:, -1]
        chunks = [z[i:i + 1024] for i in range(0, z.shape[0], 1024)]
        if self.training and torch.is_grad_enabled():
            last = torch.cat([checkpoint(run, c, use_reentrant=False) for c in chunks])
        else:
            last = torch.cat([run(c) for c in chunks])
        y = self.head(last).reshape(b, p, self.pred_len).permute(0, 2, 1)   # [B, H, P]
        return y * std + mean
