# Engineering Decisions

## D-001: In-batch negatives over explicit negative sampling

**Status:** Accepted

**Context.** For contrastive training we need negatives. Two main options:
1. Explicit negative sampling: for each (user, pos_item), sample N negatives from the item pool.
2. In-batch negatives: for a batch of (user, pos_item) pairs, every other item in the batch is a negative for this user.

**Decision.** In-batch negatives with LogQ correction.

**Why.**
- In-batch negatives are essentially free — the embeddings are already computed for the batch.
- A batch of size 1024 gives you 1023 negatives per positive. Explicit sampling at the same negative count costs N+1 item-tower forward passes per example.
- The popularity bias problem (popular items appear in batches more often, get pushed down more) has a clean fix: subtract log(item_freq) from the logits. The LogQ correction.

**When you'd choose explicit negatives.** If your batch can't fit enough items to provide signal (very small models, very small batches) or if you want curriculum learning over negative difficulty (start easy, get harder). Neither applies here.

---

## D-002: Cosine similarity (L2-normalized outputs)

**Status:** Accepted

**Context.** Two-tower scoring can be raw dot product, cosine similarity, or Euclidean distance.

**Decision.** L2-normalize both tower outputs, then dot product (= cosine similarity).

**Why.**
- Bounded outputs play nicely with softmax — no numerical issues with extreme logits.
- FAISS's inner-product index then computes cosine similarity directly.
- Raw dot product without normalization let the model "win" by inflating embedding norms instead of learning meaningful directions. We measured this: unnormalized model had higher training loss reduction but worse NDCG@10.

**Tradeoff.** Loses some expressive power (only the direction of the embedding matters, not magnitude). For retrieval at this scale, that's the right tradeoff.

---

## D-003: Temporal train/test split, not random

**Status:** Accepted

**Context.** Default `sklearn.train_test_split` is random.

**Decision.** Per-user, hold out the most recent N% of interactions.

**Why.** Random splits leak future information. If a user watches movie A then B then C, and we randomly hold out B, the model trains on "this user later watched C" and we evaluate "predict B." That's cheating. Temporal splits prevent it.

This change alone moved our Recall@10 from a misleading 0.31 down to a realistic 0.16. The 0.31 was an artifact of leakage.

**Universal lesson.** Recsys benchmarks are full of papers that quote big numbers from random splits. Be suspicious.

---

## D-004: FAISS HNSW over IVF-PQ

**Status:** Accepted for this scale

**Context.** FAISS has many index types. For our 4K items (MovieLens-1M) the choice is mostly academic, but the README claims FAISS HNSW scales to millions so we want to test that claim.

**Decision.** HNSW with M=32, efConstruction=200.

**Why.**
- HNSW is a graph index; queries traverse the graph to find neighbors. Recall is high (>0.99 of brute force on this dataset).
- IVF-PQ is better when memory is constrained (it quantizes vectors). At our scale, no memory pressure.
- Flat index (brute force) is fine up to ~1M items on a laptop. Above that, HNSW wins.

**When you'd switch.** ≥10M items + memory pressure: IVF-PQ. Distributed FAISS (FAISS doesn't natively shard; use Milvus or Vespa).

---

## D-005: MMR reranker, not learned diversification

**Status:** Accepted

**Context.** Recommenders concentrate on popular items by default. We want diversity in the top-K.

**Decision.** Maximal Marginal Relevance (MMR) as a post-retrieval rerank step. Cheap, deterministic, parameterized by one alpha knob.

**Why.** Learned diversity (e.g. DPP-based, RL-style) needs training data on what "diverse" means in your domain. For an open-source baseline, MMR with cosine distance is a reasonable starting point that doesn't require labels.

**Tradeoff.** MMR's notion of diversity is geometric (cosine distance in embedding space). It doesn't know about genres, languages, etc. — those would need feature-level diversity, which is a more complex implementation.

---

## D-006: Cold-start fallback is popularity, not metadata

**Status:** Accepted with known limitation

**Context.** New users have no embedding. We need to recommend something.

**Decision.** Top-popularity items as fallback.

**Why.** It's the strongest content-free baseline. Any user, regardless of preferences, has a non-trivial probability of liking the most popular items. Better than random.

**Known limitation.** A real recsys would use any available cold-start features (demographics, device, time, geo) to do better than global popularity. The infrastructure for that lives in the user tower's feature engineering, which is out of scope for this baseline. The cold-start fallback should be replaced with feature-based cold-start in production.

---

## R-001: Reverted — explicit negative sampling with hard negatives

I started with explicit negative sampling using "hard" negatives (items the user almost watched but didn't). The idea: train on the hard cases, model learns finer distinctions.

**Why I reverted.** The hard-negative mining itself required computing scores for many candidate items — same cost as in-batch negatives but without the elegance. Quality was marginally better but training was 3x slower. In-batch negatives + LogQ correction was Pareto-better.

## R-002: Reverted — Annoy instead of FAISS

Tried Annoy first because it's simpler to package (no compiled FAISS dep on macOS for a while). FAISS HNSW had ~2x better recall at the same query latency on this data. The packaging hassle ended up being worth it.
