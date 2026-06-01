"""Regression model.

A single Spatial-Transformer regressor that consumes a stack of visual
modality features and an optional 2-D coordinate vector. Setting
``use_geo=False`` removes the spatial branch entirely so the same class
serves both the spatial and non-spatial baselines (replacing the
duplicated ``token_reg_non_spatial.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """MSE that ignores NaN entries in ``y_true``.

    Labels are often missing for some (city, indicator) pairs and we
    don't want them to contribute to the gradient.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(y_true)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        return ((y_pred[mask] - y_true[mask]) ** 2).mean()


class SpatialTransformerRegressor(nn.Module):
    """Transformer over per-modality visual tokens, with an optional geo token.

    Args:
        input_dim: feature dimension per modality.
        num_heads: attention heads.
        num_layers: transformer encoder layers.
        hidden_dim: model width.
        output_dim: number of regression targets.
        dropout: dropout before the prediction head.
        use_geo: if True, concatenate a 2-D-projected coordinate token
            to the sequence (this is the spatially-informed variant).
            If False, the geo branch is omitted entirely.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.5,
        use_geo: bool = True,
    ):
        super().__init__()
        self.use_geo = use_geo

        self.visual_embedding = nn.Linear(input_dim, hidden_dim)

        if use_geo:
            # Lightweight MLP turning (lat, lon) into a hidden-dim "geo token".
            self.geo_mlp = nn.Sequential(
                nn.Linear(2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, n_modalities, input_dim)
        # coords: (batch, 2)  (optional, ignored when use_geo=False)
        x_emb = self.visual_embedding(x)  # -> (batch, n_modalities, hidden)

        if self.use_geo:
            assert coords is not None, "coords required when use_geo=True"
            geo_emb = self.geo_mlp(coords).unsqueeze(1)  # -> (batch, 1, hidden)
            seq = torch.cat([x_emb, geo_emb], dim=1)
        else:
            seq = x_emb

        out = self.transformer(seq)
        out = out.mean(dim=1)  # token-mean pooling
        out = self.dropout(out)
        return self.fc(out)
