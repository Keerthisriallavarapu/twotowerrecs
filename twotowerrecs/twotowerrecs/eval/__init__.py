"""Offline evaluation metrics: Recall@k, NDCG@k, catalog coverage.

Recall@k = (1 if any positive item appears in top-k) — averaged over users.
NDCG@k = discounted gain normalized to [0,1]. Captures position sensitivity.
Catalog coverage = fraction of items that appear in ANY user's top-k. Captures
whether the model just recommends the same 10 popular items to everyone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    recall_at_k: float
    ndcg_at_k: float
    catalog_coverage: float
    n_users_evaluated: int
    k: int


def recall_at_k(true_items: set[int], pred_items: list[int], k: int) -> float:
    """Hit-rate flavor: 1 if any true item in top-k, else 0."""
    return 1.0 if set(pred_items[:k]) & true_items else 0.0


def ndcg_at_k(true_items: set[int], pred_items: list[int], k: int) -> float:
    """NDCG with binary relevance. Each true item in top-k contributes
    1/log2(rank+2). Ideal DCG assumes all true items are at the top."""
    dcg = 0.0
    for rank, item in enumerate(pred_items[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(rank + 2)
    # IDCG: best case is all true items consecutively from rank 0
    n_relevant = min(len(true_items), k)
    if n_relevant == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(r + 2) for r in range(n_relevant))
    return dcg / idcg


def evaluate(
    server,  # TwoTowerServer
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
) -> EvalResult:
    """Evaluate the served model on a held-out test set.

    test_df: held-out (user, item) interactions
    train_df: training interactions (used to exclude seen items from recs)
    """
    # Group test items per user
    test_by_user = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    seen_by_user = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    user_ids = list(test_by_user.keys())
    if max_users:
        user_ids = user_ids[:max_users]

    recalls = []
    ndcgs = []
    all_recommended_items = set()

    for uid in tqdm(user_ids, desc=f"Evaluating @{k}"):
        true_items = test_by_user[uid]
        seen = seen_by_user.get(uid, set())
        recs = server.recommend(uid, k=k, exclude_seen=seen)
        pred = [r.movie_idx for r in recs]
        recalls.append(recall_at_k(true_items, pred, k))
        ndcgs.append(ndcg_at_k(true_items, pred, k))
        all_recommended_items.update(pred)

    return EvalResult(
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        catalog_coverage=len(all_recommended_items) / server.n_items,
        n_users_evaluated=len(recalls),
        k=k,
    )
