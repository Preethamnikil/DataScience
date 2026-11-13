"""Cached loaders for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#c0392b",
    "neutral": "#7f8c8d",
    "positive": "#1e8449",
}


def load_summary() -> dict:
    return json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "metrics.csv")


def load_labeled_sample(n: int = 5000) -> pd.DataFrame:
    from src.preprocess import clean_tweet, map_label

    df = pd.read_csv(DATA_DIR / "Twitter_Data.csv")
    df = df.dropna(subset=["clean_text", "category"]).copy()
    df["text"] = df["clean_text"].astype(str)
    df["label"] = df["category"].map(map_label)
    df = df.dropna(subset=["label"])
    df["clean_text"] = df["text"].map(clean_tweet)
    df = df[df["clean_text"].str.split().str.len() >= 3]
    df["leader"] = np.where(
        df["text"].str.contains(r"\brahul\b", case=False, regex=True),
        "Rahul Gandhi",
        np.where(df["text"].str.contains(r"\bmodi\b", case=False, regex=True), "Narendra Modi", "Other"),
    )
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    return df[["text", "clean_text", "label", "leader"]].reset_index(drop=True)


def load_svm_bundle():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS_DIR / "svm.joblib")
    return vectorizer, model


def figure_path(name: str) -> Path | None:
    path = FIGURES_DIR / name
    return path if path.exists() else None


def load_state_summary() -> dict:
    path = RESULTS_DIR / "state_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            "State summary not found. Run: py -3 build_state_data.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_state_tweets(state: str | None = None) -> pd.DataFrame:
    path = RESULTS_DIR / "state_tweets.csv"
    if not path.exists():
        raise FileNotFoundError(
            "State tweets not found. Run: py -3 build_state_data.py"
        )
    df = pd.read_csv(path)
    if state:
        df = df[df["state"] == state]
    return df
