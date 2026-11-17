"""User feedback storage for Trouvaille (SQLite, no extra dependencies)."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "feedback.db"

# How strongly feedback can move a product's rank (same scale as tone/fit: 0.4).
MAX_BOOST = 0.2
# Smoothing: a product needs several votes before feedback matters much.
SMOOTHING = 5


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP,
               product_url TEXT NOT NULL,
               title TEXT,
               brand TEXT,
               occasion TEXT NOT NULL,
               vote INTEGER NOT NULL CHECK (vote IN (-1, 1))
           )"""
    )
    return conn


def add_feedback(product_url, title, brand, occasion, vote):
    """Save one vote. vote = 1 (helpful) or -1 (not helpful)."""
    if vote not in (1, -1):
        raise ValueError("vote must be 1 or -1")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (product_url, title, brand, occasion, vote) VALUES (?, ?, ?, ?, ?)",
            (product_url, title, brand, occasion, vote),
        )


def load_feedback():
    """All votes as a DataFrame (empty if none yet)."""
    with _connect() as conn:
        return pd.read_sql_query("SELECT * FROM feedback", conn)


def feedback_boosts(occasion):
    """Return {product_url: rank adjustment} for one occasion.

    adjustment = MAX_BOOST * (ups - downs) / (ups + downs + SMOOTHING)
    so it stays between -MAX_BOOST and +MAX_BOOST and grows slowly with votes.
    """
    fb = load_feedback()
    fb = fb[fb["occasion"] == occasion]
    if fb.empty:
        return {}
    g = fb.groupby("product_url")["vote"].agg(
        ups=lambda v: (v == 1).sum(), downs=lambda v: (v == -1).sum()
    )
    boost = MAX_BOOST * (g["ups"] - g["downs"]) / (g["ups"] + g["downs"] + SMOOTHING)
    return boost.to_dict()


def occasion_summary():
    """Approval rate and vote count per occasion."""
    fb = load_feedback()
    if fb.empty:
        return pd.DataFrame(columns=["occasion", "votes", "approval"])
    out = fb.groupby("occasion")["vote"].agg(
        votes="count", approval=lambda v: (v == 1).mean()
    )
    return out.reset_index()


def top_products(n=5, best=True):
    """Products with the best (or worst) net score."""
    fb = load_feedback()
    if fb.empty:
        return pd.DataFrame(columns=["title", "brand", "net_score", "votes"])
    g = fb.groupby(["product_url", "title", "brand"])["vote"].agg(
        net_score="sum", votes="count"
    ).reset_index()
    g = g.sort_values("net_score", ascending=not best).head(n)
    return g[["title", "brand", "net_score", "votes"]]