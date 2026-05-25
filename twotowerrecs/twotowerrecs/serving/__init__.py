"""Serving: FAISS index for ANN retrieval + a small re-ranking layer.

Architecture:
1. Item tower embeddings -> FAISS index (we use HNSW for the speed/quality
   tradeoff that fits this scale; for >10M items consider IVF-PQ).
2. At request time: encode user_id with the user tower -> query FAISS for
   top-K candidates -> apply a diversity reranker -> return.

For cold-start users (no embedding), we serve a popularity-based fallback.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from ..models import TowerConfig, TwoTowerModel

log = logging.getLogger(__name__)


@dataclass
class Recommendation:
    movie_idx: int
    score: float
    rank: int


class TwoTowerServer:
    def __init__(
        self,
        artifacts_dir: str | Path,
        movies_table: pd.DataFrame | None = None,
        device: str | None = None,
    ):
        artifacts_dir = Path(artifacts_dir)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint to recover the config
        ckpt = torch.load(artifacts_dir / "final.pt", map_location=device)
        # checkpoint_epochN.pt has more metadata; final.pt is just state_dict.
        # We need n_users/n_items from somewhere. Conventionally we look at
        # the latest checkpoint file.
        meta_ckpts = sorted(artifacts_dir.glob("checkpoint_epoch*.pt"))
        if meta_ckpts:
            meta = torch.load(meta_ckpts[-1], map_location="cpu")
            n_users = meta["n_users"]
            n_items = meta["n_items"]
            cfg = meta["config"]
        else:
            raise FileNotFoundError("No checkpoint_epochN.pt found; can't recover config")

        user_cfg = TowerConfig(
            n_entities=n_users,
            embedding_dim=cfg["embedding_dim"],
            output_dim=cfg["tower_output_dim"],
        )
        item_cfg = TowerConfig(
            n_entities=n_items,
            embedding_dim=cfg["embedding_dim"],
            output_dim=cfg["tower_output_dim"],
        )
        self.model = TwoTowerModel(user_cfg, item_cfg).to(device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = cfg["tower_output_dim"]

        # Load item embeddings and build FAISS index
        item_embs = np.load(artifacts_dir / "item_embeddings.npy").astype(np.float32)
        assert item_embs.shape == (n_items, self.embed_dim)
        self.item_embs = item_embs
        self.index = self._build_index(item_embs)
        log.info("Loaded FAISS index: %d items, dim=%d", n_items, self.embed_dim)

        self.movies = movies_table
        # Cold-start fallback: popularity (just movie_idx 0..K — caller supplies
        # popularity at production; demo uses index order).
        self._fallback = np.arange(min(100, n_items))

    @staticmethod
    def _build_index(item_embs: np.ndarray) -> faiss.Index:
        """HNSW index with inner-product metric. M=32 is a balanced choice;
        larger M = better recall at higher memory."""
        dim = item_embs.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(item_embs)
        return index

    @torch.inference_mode()
    def encode_user(self, user_idx: int) -> np.ndarray:
        if not (0 <= user_idx < self.n_users):
            raise ValueError(f"user_idx {user_idx} out of range [0, {self.n_users})")
        ids = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        emb = self.model.user_tower(ids).cpu().numpy().astype(np.float32)
        return emb  # (1, D)

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: set[int] | None = None,
        diversity_alpha: float = 0.0,
    ) -> list[Recommendation]:
        """Top-k recs for a user.

        exclude_seen: movie_idxs the user has already interacted with.
        diversity_alpha: 0.0 = pure relevance, 1.0 = maximal diversity (MMR).
        """
        try:
            query = self.encode_user(user_idx)
        except ValueError:
            log.info("Cold-start fallback for user_idx=%s", user_idx)
            return [
                Recommendation(movie_idx=int(m), score=0.0, rank=r)
                for r, m in enumerate(self._fallback[:k])
            ]

        # Fetch more than k so we can re-rank and filter seen items
        oversample_k = max(k * 4, 50)
        scores, ids = self.index.search(query, oversample_k)
        scores, ids = scores[0], ids[0]

        # Filter seen
        if exclude_seen:
            keep_mask = np.array([i not in exclude_seen for i in ids])
            scores = scores[keep_mask]
            ids = ids[keep_mask]

        # Optional diversity reranking (MMR-lite)
        if diversity_alpha > 0:
            ids, scores = self._mmr_rerank(query, ids, scores, k, diversity_alpha)
        else:
            ids = ids[:k]
            scores = scores[:k]

        return [
            Recommendation(movie_idx=int(m), score=float(s), rank=r)
            for r, (m, s) in enumerate(zip(ids, scores, strict=True))
        ]

    def _mmr_rerank(
        self,
        query: np.ndarray,  # (1, D)
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
        k: int,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Maximal Marginal Relevance.
        relevance = candidate_score (already a dot product with query)
        diversity = -max similarity to already-selected items
        final = (1-alpha) * relevance + alpha * diversity
        """
        candidate_embs = self.item_embs[candidate_ids]
        selected_ids: list[int] = []
        selected_scores: list[float] = []
        selected_mask = np.zeros(len(candidate_ids), dtype=bool)

        while len(selected_ids) < k and not selected_mask.all():
            if not selected_ids:
                # First pick is highest-scoring
                pick = int(np.argmax(candidate_scores))
            else:
                selected_embs = self.item_embs[selected_ids]
                # Similarity of each candidate to selected set
                sims = candidate_embs @ selected_embs.T  # (C, S)
                max_sim = sims.max(axis=1)
                mmr = (1 - alpha) * candidate_scores - alpha * max_sim
                mmr[selected_mask] = -np.inf
                pick = int(np.argmax(mmr))
            selected_mask[pick] = True
            selected_ids.append(int(candidate_ids[pick]))
            selected_scores.append(float(candidate_scores[pick]))

        return np.array(selected_ids), np.array(selected_scores)


def benchmark_p99(server: TwoTowerServer, n_queries: int = 1000, k: int = 10) -> dict:
    """Quick latency benchmark."""
    rng = np.random.default_rng(0)
    user_ids = rng.integers(0, server.n_users, size=n_queries)
    latencies = []
    for u in user_ids:
        start = time.perf_counter()
        server.recommend(int(u), k=k)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    return {
        "n": n_queries,
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": latencies[n_queries // 2],
        "p99_ms": latencies[int(n_queries * 0.99)],
        "p99_9_ms": latencies[int(n_queries * 0.999)] if n_queries >= 1000 else latencies[-1],
    }
