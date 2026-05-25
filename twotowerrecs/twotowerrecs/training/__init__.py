"""Training loop for the two-tower model.

Key training details that matter:
- Mixed precision (bf16 on Ampere+, fp16 on Volta-Turing). Cuts training time
  ~2x for free.
- Large batches help in-batch negative sampling — more in-batch negatives
  per positive means a harder, more useful contrastive task.
- Compute item frequencies once over the training set for LogQ correction.
- Save the item embeddings after every epoch; the FAISS index will be built
  from the latest embeddings.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models import TowerConfig, TwoTowerModel, in_batch_loss

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 5
    embedding_dim: int = 64
    tower_output_dim: int = 32
    temperature: float = 0.05
    use_amp: bool = True
    num_workers: int = 4
    seed: int = 42


class InteractionDataset(Dataset):
    """Positive (user, item) pairs from the interactions dataframe."""

    def __init__(self, interactions: pd.DataFrame):
        self.users = torch.tensor(interactions["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(interactions["movie_idx"].values, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        return self.users[i], self.items[i]


def compute_item_freqs(interactions: pd.DataFrame, n_items: int) -> torch.Tensor:
    """Empirical probability of each item being sampled. Used for LogQ."""
    counts = Counter(interactions["movie_idx"].values.tolist())
    total = sum(counts.values())
    freqs = np.zeros(n_items, dtype=np.float32)
    for idx, c in counts.items():
        freqs[idx] = c / total
    return torch.from_numpy(freqs)


def train(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    config: TrainConfig,
    out_dir: str | Path,
    device: str | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s with %d interactions", device, len(train_df))

    # Build the model
    user_cfg = TowerConfig(
        n_entities=n_users,
        embedding_dim=config.embedding_dim,
        output_dim=config.tower_output_dim,
    )
    item_cfg = TowerConfig(
        n_entities=n_items,
        embedding_dim=config.embedding_dim,
        output_dim=config.tower_output_dim,
    )
    model = TwoTowerModel(user_cfg, item_cfg).to(device)
    log.info("Model has %d params", sum(p.numel() for p in model.parameters()))

    # Optimizer + AMP
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device=device) if config.use_amp else None
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Data + item-frequency table for LogQ
    dataset = InteractionDataset(train_df)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    item_freqs = compute_item_freqs(train_df, n_items).to(device)

    history = {"epoch_loss": []}
    for epoch in range(config.n_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.n_epochs}")
        for user_ids, item_ids in pbar:
            user_ids = user_ids.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            batch_freqs = item_freqs[item_ids]

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                    u_emb, i_emb = model(user_ids, item_ids)
                    loss = in_batch_loss(u_emb, i_emb, config.temperature, batch_freqs)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                u_emb, i_emb = model(user_ids, item_ids)
                loss = in_batch_loss(u_emb, i_emb, config.temperature, batch_freqs)
                loss.backward()
                opt.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / max(n_batches, 1)
        history["epoch_loss"].append(epoch_loss)
        log.info(
            "Epoch %d done: loss=%.4f time=%.1fs",
            epoch + 1, epoch_loss, time.time() - epoch_start,
        )

        # Save checkpoint every epoch — easy to recover from bad runs
        torch.save({
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "epoch": epoch + 1,
            "n_users": n_users,
            "n_items": n_items,
        }, out_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Final artifacts: model and pre-computed item embeddings for serving
    torch.save(model.state_dict(), out_dir / "final.pt")
    log.info("Computing item embeddings for FAISS index...")
    model.eval()
    with torch.inference_mode():
        all_item_ids = torch.arange(n_items, device=device)
        item_embs = []
        chunk = 8192
        for i in range(0, n_items, chunk):
            chunk_ids = all_item_ids[i:i + chunk]
            item_embs.append(model.item_tower(chunk_ids).cpu())
        item_embs = torch.cat(item_embs, dim=0).numpy().astype(np.float32)
    np.save(out_dir / "item_embeddings.npy", item_embs)
    log.info("Saved item embeddings: shape=%s", item_embs.shape)

    return history
