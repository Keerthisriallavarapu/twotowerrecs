"""Tests for the two-tower model and loss."""
from __future__ import annotations

import torch

from twotowerrecs.models import TowerConfig, TwoTowerModel, in_batch_loss


def test_model_forward_shapes():
    n_users = 100
    n_items = 200
    m = TwoTowerModel(
        TowerConfig(n_entities=n_users, output_dim=32),
        TowerConfig(n_entities=n_items, output_dim=32),
    )
    u = torch.randint(0, n_users, (8,))
    i = torch.randint(0, n_items, (8,))
    u_emb, i_emb = m(u, i)
    assert u_emb.shape == (8, 32)
    assert i_emb.shape == (8, 32)


def test_outputs_are_unit_norm():
    m = TwoTowerModel(
        TowerConfig(n_entities=100, output_dim=16),
        TowerConfig(n_entities=200, output_dim=16),
    )
    u, _ = m(torch.arange(5), torch.arange(5))
    norms = u.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_in_batch_loss_is_nonnegative():
    batch = 16
    dim = 8
    u_emb = torch.nn.functional.normalize(torch.randn(batch, dim), p=2, dim=-1)
    i_emb = torch.nn.functional.normalize(torch.randn(batch, dim), p=2, dim=-1)
    loss = in_batch_loss(u_emb, i_emb)
    assert loss.item() >= 0
    assert torch.isfinite(loss).item()


def test_in_batch_loss_low_when_aligned():
    """If user and item embeddings on the diagonal are identical and
    everything else is orthogonal, loss should be very low."""
    batch = 16
    dim = 8
    # Make diagonal pairs identical, off-diagonal random orthogonal
    pairs = torch.nn.functional.normalize(torch.randn(batch, dim), p=2, dim=-1)
    loss = in_batch_loss(pairs, pairs, temperature=0.05)
    # With low temp and aligned pairs, softmax is sharp -> loss small
    assert loss.item() < 1.0


def test_logq_correction_changes_loss():
    batch = 32
    dim = 16
    u = torch.nn.functional.normalize(torch.randn(batch, dim), p=2, dim=-1)
    i = torch.nn.functional.normalize(torch.randn(batch, dim), p=2, dim=-1)
    freqs = torch.softmax(torch.randn(batch), dim=0)
    loss_no_correction = in_batch_loss(u, i)
    loss_corrected = in_batch_loss(u, i, item_freq=freqs)
    # They should differ (not asserting direction, just that correction does something)
    assert not torch.isclose(loss_no_correction, loss_corrected, atol=1e-3)
