"""Tests for eval metrics."""
from __future__ import annotations

import numpy as np

from twotowerrecs.eval import ndcg_at_k, recall_at_k


def test_recall_hit():
    assert recall_at_k({1, 2}, [5, 1, 7, 9], k=10) == 1.0


def test_recall_miss():
    assert recall_at_k({1, 2}, [5, 6, 7, 9], k=10) == 0.0


def test_recall_no_true_items():
    # No relevant items -> 0 (no signal possible)
    assert recall_at_k(set(), [1, 2, 3], k=10) == 0.0


def test_ndcg_perfect_ordering():
    # All true items at the top
    true = {1, 2, 3}
    pred = [1, 2, 3, 4, 5, 6]
    assert ndcg_at_k(true, pred, k=5) == 1.0


def test_ndcg_worst_ordering():
    # No true items in top-k -> 0
    true = {7, 8, 9}
    pred = [1, 2, 3, 4, 5]
    assert ndcg_at_k(true, pred, k=5) == 0.0


def test_ndcg_partial():
    # Mixed in: first item is good, then misses, then good
    true = {1, 3}
    pred = [1, 5, 3, 7, 9]
    val = ndcg_at_k(true, pred, k=5)
    # Should be between 0 and 1 and finite
    assert 0 < val < 1
    assert np.isfinite(val)
