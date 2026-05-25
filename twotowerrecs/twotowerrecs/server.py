"""HTTP API for the recommender."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .serving import TwoTowerServer

log = logging.getLogger(__name__)


class RecRequest(BaseModel):
    user_idx: int
    k: int = 10
    diversity_alpha: float = 0.0


class RecItem(BaseModel):
    movie_idx: int
    title: str | None = None
    genres: str | None = None
    score: float
    rank: int


class RecResponse(BaseModel):
    user_idx: int
    items: list[RecItem]
    served_by: str  # "tower" | "fallback"


def create_app(artifacts_dir: str | Path, movies_path: str | Path | None = None) -> FastAPI:
    server = TwoTowerServer(artifacts_dir=artifacts_dir)
    movies: pd.DataFrame | None = None
    if movies_path is not None and Path(movies_path).exists():
        movies = pd.read_parquet(movies_path)
        # Index by movie_idx for fast lookup
        movies = movies.set_index("movie_idx")
        log.info("Loaded movies table with %d rows", len(movies))

    app = FastAPI(title="TwoTowerRecs", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "n_users": server.n_users, "n_items": server.n_items}

    @app.post("/recommend", response_model=RecResponse)
    def recommend(req: RecRequest):
        try:
            recs = server.recommend(
                req.user_idx, k=req.k, diversity_alpha=req.diversity_alpha
            )
        except Exception as e:
            raise HTTPException(500, str(e)) from e

        items = []
        for r in recs:
            title = genres = None
            if movies is not None and r.movie_idx in movies.index:
                row = movies.loc[r.movie_idx]
                title = str(row.get("title"))
                genres = str(row.get("genres"))
            items.append(RecItem(
                movie_idx=r.movie_idx,
                title=title,
                genres=genres,
                score=r.score,
                rank=r.rank,
            ))
        return RecResponse(
            user_idx=req.user_idx,
            items=items,
            served_by="tower" if req.user_idx < server.n_users else "fallback",
        )

    return app
