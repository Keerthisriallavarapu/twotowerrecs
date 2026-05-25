"""Two-tower retrieval model.

Two MLPs (towers): one for users, one for items. Each produces an embedding
of the same dimension. Score = dot product of user_emb and item_emb.

Training: in-batch negative sampling. For a batch of (user, positive_item)
pairs, every other item in the batch is treated as a negative for this user.
This is far more efficient than explicit negative sampling and works well
because positives are rare in the batch.

The catch: in-batch negatives are biased toward popular items (popular items
appear in batches more often, get sampled as negatives more often, learn
to score lower). The fix is the LogQ correction in `bias_corrected_loss`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TowerConfig:
    n_entities: int             # n_users or n_items
    embedding_dim: int = 64     # the table embedding dim
    hidden_dims: tuple[int, ...] = (128, 64)
    output_dim: int = 32        # final tower output dim — must match across towers
    dropout: float = 0.1


class Tower(nn.Module):
    """A single tower: ID embedding -> MLP -> L2-normalized output."""

    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.n_entities, cfg.embedding_dim)

        dims = [cfg.embedding_dim, *cfg.hidden_dims, cfg.output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
        # Drop the last ReLU/Dropout so the output is unbounded; we'll L2-norm
        layers = layers[:-2]
        self.mlp = nn.Sequential(*layers)

        # Better embedding init than the default (avoids small grads early)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.01)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)  # cosine similarity = dot product


class TwoTowerModel(nn.Module):
    def __init__(self, user_cfg: TowerConfig, item_cfg: TowerConfig):
        super().__init__()
        assert user_cfg.output_dim == item_cfg.output_dim, \
            "Towers must output same dim"
        self.user_tower = Tower(user_cfg)
        self.item_tower = Tower(item_cfg)
        self.embedding_dim = user_cfg.output_dim

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.item_tower(item_ids)


def in_batch_loss(
    user_emb: torch.Tensor,  # (B, D)
    item_emb: torch.Tensor,  # (B, D)
    temperature: float = 0.05,
    item_freq: torch.Tensor | None = None,  # (B,) — for LogQ correction
) -> torch.Tensor:
    """Sampled softmax with in-batch negatives.

    item_freq is the prior probability of each batch item being sampled.
    If provided, applies LogQ correction (subtracts log(freq) from logits
    so popular items aren't over-penalized).

    Reference: "Sampling-Bias-Corrected Neural Modeling for Large Corpus
    Item Recommendations" (Yi et al. 2019).
    """
    # Logits: (B, B). Each row is one user's scores against all items in batch.
    logits = (user_emb @ item_emb.t()) / temperature

    if item_freq is not None:
        # Subtract log of sampling prob to debias. We expand freq to (1, B)
        # so it broadcasts across users.
        log_freq = torch.log(item_freq.clamp(min=1e-12)).unsqueeze(0)
        logits = logits - log_freq

    # Positives are on the diagonal: user_i's positive is item_i
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def cosine_similarity_batch(
    queries: torch.Tensor,  # (Q, D)
    candidates: torch.Tensor,  # (C, D)
) -> torch.Tensor:
    """For evaluation: similarity scores between Q queries and C candidates."""
    q = F.normalize(queries, p=2, dim=-1)
    c = F.normalize(candidates, p=2, dim=-1)
    return q @ c.t()
