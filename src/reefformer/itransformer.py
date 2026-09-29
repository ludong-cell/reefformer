"""Compact checkpoint-compatible iTransformer implementation.

The architecture and module names follow the MIT-licensed THUML iTransformer
implementation at commit c2426e68ca13f74aaec08045c5c724d8ad328124. Only the
components needed by the archived encoder-only checkpoints are retained.
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


class DataEmbeddingInverted(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class FullAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attention, values).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = FullAttention(dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        batch, length, _ = queries.shape
        heads = self.n_heads
        q = self.query_projection(queries).view(batch, length, heads, -1)
        k = self.key_projection(queries).view(batch, length, heads, -1)
        v = self.value_projection(queries).view(batch, length, heads, -1)
        output = self.inner_attention(q, k, v).view(batch, length, -1)
        return self.out_projection(output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float, activation: str):
        super().__init__()
        self.attention = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayer], d_model: int):
        super().__init__()
        self.attn_layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.attn_layers:
            x = layer(x)
        return self.norm(x)


class ITransformer(nn.Module):
    """Encoder-only iTransformer used in the manuscript experiments."""

    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = int(config["seq_len"])
        self.pred_len = int(config["pred_len"])
        self.use_norm = bool(config["use_norm"])
        d_model = int(config["d_model"])
        dropout = float(config["dropout"])
        self.enc_embedding = DataEmbeddingInverted(self.seq_len, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_ff=int(config["d_ff"]),
                    n_heads=int(config["n_heads"]),
                    dropout=dropout,
                    activation=str(config["activation"]),
                )
                for _ in range(int(config["e_layers"]))
            ],
            d_model=d_model,
        )
        self.projector = nn.Linear(d_model, self.pred_len, bias=True)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        spatial_units = x_enc.shape[2]
        encoded = self.encoder(self.enc_embedding(x_enc))
        output = self.projector(encoded).permute(0, 2, 1)[:, :, :spatial_units]
        if self.use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            output = output + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return output
