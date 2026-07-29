"""Encoder Espacial-Temporal de landmarks (do zero)."""

import torch.nn as nn

from skeltrans.common.layout import INPUT_DIM
from skeltrans.training.models.positional_encoding import SinusoidalPositionalEncoding


class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.2, out_dim=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # Conv1D temporal (kernel=3) preservando o comprimento T
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        # adaptador para o hidden do decoder (T5), se necessario
        out_dim = out_dim or d_model
        self.adapter = nn.Identity() if out_dim == d_model else nn.Linear(d_model, out_dim)

    def forward(self, feats, pad_mask):
        """feats: (B, T, INPUT_DIM); pad_mask: (B, T) True onde e padding."""
        x = self.input_proj(feats)                 # (B, T, d_model)
        x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)  # conv sobre o tempo
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)     # (B, T, d_model)
        return self.adapter(x)                      # (B, T, out_dim)
