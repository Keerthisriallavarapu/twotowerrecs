"""CLI: download data, train, evaluate, serve."""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command(name="train")
def train_cmd(
    variant: str = typer.Option("1m", help="MovieLens variant: 1m or 25m"),
    epochs: int = 5,
    batch_size: int = 1024,
    embedding_dim: int = 64,
    out_dir: Path = Path("artifacts"),
):
    """Download data and train the two-tower model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    from .data import load_movielens, train_test_split_temporal
    from .training import TrainConfig, train

    data = load_movielens(variant=variant)
    console.print(f"[green]Loaded {data.n_interactions} interactions, "
                  f"{data.n_users} users, {data.n_movies} movies[/green]")

    train_df, test_df = train_test_split_temporal(data.interactions, test_frac=0.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(out_dir / "test_interactions.parquet")
    data.movies.to_parquet(out_dir / "movies.parquet")

    cfg = TrainConfig(
        batch_size=batch_size,
        n_epochs=epochs,
        embedding_dim=embedding_dim,
    )
    history = train(
        train_df=train_df,
        n_users=data.n_users,
        n_items=data.n_movies,
        config=cfg,
        out_dir=out_dir,
    )
    console.print(f"[green]Training done. Final loss: {history['epoch_loss'][-1]:.4f}[/green]")


@app.command(name="eval")
def eval_cmd(
    artifacts: Path = Path("artifacts"),
    k: int = 10,
    max_users: int = 5000,
):
    """Run offline evaluation on the held-out test set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    import pandas as pd
    from .eval import evaluate
    from .serving import TwoTowerServer

    test_df = pd.read_parquet(artifacts / "test_interactions.parquet")
    # The train set we need to exclude seen items — reconstruct from parquet
    # if available, otherwise just don't exclude (gives a lower bound on metrics).
    train_df_path = artifacts / "train_interactions.parquet"
    if train_df_path.exists():
        train_df = pd.read_parquet(train_df_path)
    else:
        train_df = pd.DataFrame(columns=["user_idx", "movie_idx"])

    server = TwoTowerServer(artifacts_dir=artifacts)
    result = evaluate(server, test_df, train_df, k=k, max_users=max_users)

    table = Table(title=f"Eval results @k={k}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Recall@k", f"{result.recall_at_k:.4f}")
    table.add_row("NDCG@k", f"{result.ndcg_at_k:.4f}")
    table.add_row("Catalog coverage", f"{result.catalog_coverage:.4f}")
    table.add_row("Users evaluated", str(result.n_users_evaluated))
    console.print(table)


@app.command(name="serve")
def serve_cmd(
    artifacts: Path = Path("artifacts"),
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """Start the recommender HTTP API."""
    import uvicorn

    from .server import create_app
    movies = artifacts / "movies.parquet"
    movies_path = str(movies) if movies.exists() else None

    app_ = create_app(artifacts_dir=str(artifacts), movies_path=movies_path)
    uvicorn.run(app_, host=host, port=port)


@app.command(name="bench")
def bench_cmd(
    artifacts: Path = Path("artifacts"),
    n_queries: int = 1000,
):
    """Quick latency benchmark."""
    from .serving import TwoTowerServer, benchmark_p99

    server = TwoTowerServer(artifacts_dir=artifacts)
    result = benchmark_p99(server, n_queries=n_queries)
    table = Table(title=f"Latency over {n_queries} queries")
    table.add_column("Metric")
    table.add_column("ms", justify="right")
    for k, v in result.items():
        if k == "n":
            continue
        table.add_row(k, f"{v:.2f}")
    console.print(table)


if __name__ == "__main__":
    app()
