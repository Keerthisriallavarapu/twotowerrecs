# TwoTowerRecs

Two-tower retrieval model with FAISS serving and a small UI. Trained on MovieLens.

The two-tower architecture is what Netflix, YouTube, TikTok, and Pinterest all use for the *retrieval* stage of their recommenders (the part that picks 1000 candidates out of 50M items before a heavier ranker re-orders them). This is a minimal, end-to-end implementation: data → model → ANN serving → evaluation.

## What's here

- **Two-tower model** in PyTorch with cosine-similarity scoring and in-batch negatives.
- **LogQ correction** for popularity bias in in-batch negative sampling.
- **MovieLens data pipeline**: download, temporal train/test split, integer encoding.
- **FAISS HNSW** serving with diversity reranking (MMR) and cold-start fallback.
- **Offline eval**: Recall@k, NDCG@k, catalog coverage.
- **FastAPI server** with a movie metadata join.
- **Frontend** (Next.js) for clicking around.

## Quick start

```bash
pip install -e ".[dev]"

# Train on MovieLens-1M (downloads ~6MB, trains in ~5 min on a laptop)
ttr train --variant 1m --epochs 5

# Evaluate on the held-out set
ttr eval

# Latency benchmark
ttr bench --n-queries 1000

# Serve
ttr serve
# In another shell:
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 42, "k": 10}'

# Frontend
cd frontend && pnpm install && pnpm dev
```

For ML-25M, swap `--variant 25m`. Trains in ~30 min on a single GPU.

## Why two-tower and not [your favorite alternative]

This is the structural choice production recsys teams keep returning to. Reasons in order of importance:

1. **Decoupling at serve time.** User embeddings can be computed on-demand from a query, item embeddings can be precomputed and indexed. You can index 100M items with FAISS and serve at p99 <20ms.
2. **Scales to billions of items.** Cross-architectures that score (user, item) jointly need to evaluate every (user, item) pair at retrieval. Two-tower scores via dot product, so ANN works.
3. **Composable with rerankers.** The two-tower outputs are the "retrieve K candidates" step. A heavier model reranks the K to a final order. This pattern is universal.

The model loses to joint models on quality per-item but wins by allowing you to retrieve from a much larger candidate set in the same time budget. Larger candidate set + reasonable model > smaller candidate set + sophisticated model, empirically.

## Performance

Target numbers on MovieLens-1M, training on CPU:

| Metric | Target |
|---|---|
| Recall@10 | ~0.16 |
| NDCG@10 | ~0.09 |
| Catalog coverage @10 | ~0.45 |
| Serving p99 (single-thread CPU, 1M users / 4K items) | <5ms |

Numbers here are intentionally conservative — your run on the same data should match within 10-15%. If they're much worse, check the training loss is decreasing properly. If they're much better, you've probably leaked test data into train (common bug: random splits instead of temporal).

These are the baseline numbers. Improving on them is the actual recsys job — better towers, better features, better negative sampling, better reranking. See [docs/DECISIONS.md](docs/DECISIONS.md) for what I tried that didn't work.

## Project layout

```
twotowerrecs/
├── twotowerrecs/
│   ├── data/        # MovieLens loader + temporal split
│   ├── models/      # Two-tower architecture + losses
│   ├── training/    # Train loop with AMP, LogQ correction
│   ├── serving/     # FAISS HNSW + MMR reranker + cold-start fallback
│   ├── eval/        # Recall, NDCG, coverage
│   ├── server.py    # FastAPI
│   └── cli.py
├── frontend/        # Next.js UI
├── tests/
└── docs/
```

## License

Apache 2.0.
