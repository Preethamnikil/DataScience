"""Tag tweets with Indian states and build dashboard aggregates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_tweet, map_label
from src.states import ELECTION_INFO, STATE_ORDER, detect_states

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_all_tweets() -> pd.DataFrame:
    frames = []

    labeled = pd.read_csv(DATA_DIR / "Twitter_Data.csv")
    labeled = labeled.dropna(subset=["clean_text", "category"]).copy()
    labeled["text"] = labeled["clean_text"].astype(str)
    labeled["label"] = labeled["category"].map(map_label)
    labeled = labeled.dropna(subset=["label"])
    labeled["source"] = "Twitter_Data"
    frames.append(labeled[["text", "label", "source"]])

    for fname, leader in [("Modi.csv", "Modi"), ("Rahul.csv", "Rahul")]:
        path = DATA_DIR / fname
        if path.exists():
            raw = pd.read_csv(path)
            raw["text"] = raw["Tweet"].astype(str)
            raw["label"] = None
            raw["source"] = fname
            frames.append(raw[["text", "label", "source"]])

    df = pd.concat(frames, ignore_index=True)
    df["clean_text"] = df["text"].map(clean_tweet)
    df = df[df["clean_text"].str.split().str.len() >= 3].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def build_state_rows(df: pd.DataFrame) -> pd.DataFrame:
    states_col = df["text"].map(detect_states)
    df = df.copy()
    df["states"] = states_col.map(lambda s: "|".join(s) if s else "")
    df["state_count"] = states_col.map(len)
    tagged = df[df["state_count"] > 0].copy()
    return tagged


def explode_by_state(tagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tagged.iterrows():
        for state in row["states"].split("|"):
            rows.append(
                {
                    "text": row["text"],
                    "clean_text": row["clean_text"],
                    "label": row["label"],
                    "source": row["source"],
                    "state": state,
                }
            )
    return pd.DataFrame(rows)


def summarize_states(exploded: pd.DataFrame) -> dict:
    summary = {"states": {}, "national": {}}

    labeled = exploded[exploded["label"].notna()]
    if len(labeled):
        nat_counts = labeled["label"].value_counts()
        summary["national"] = {
            "tweet_mentions": int(len(labeled)),
            "positive": int(nat_counts.get("positive", 0)),
            "neutral": int(nat_counts.get("neutral", 0)),
            "negative": int(nat_counts.get("negative", 0)),
            "positive_pct": round(nat_counts.get("positive", 0) / len(labeled), 4),
            "neutral_pct": round(nat_counts.get("neutral", 0) / len(labeled), 4),
            "negative_pct": round(nat_counts.get("negative", 0) / len(labeled), 4),
        }

    for state in STATE_ORDER:
        sub = exploded[exploded["state"] == state]
        if sub.empty:
            continue
        labeled_sub = sub[sub["label"].notna()]
        counts = labeled_sub["label"].value_counts() if len(labeled_sub) else pd.Series(dtype=int)
        total = int(len(sub))
        labeled_n = int(len(labeled_sub))
        info = ELECTION_INFO.get(state)
        summary["states"][state] = {
            "region": info.region if info else "",
            "lok_sabha_seats": info.lok_sabha_seats if info else 0,
            "election_phase": info.election_phase if info else "",
            "major_parties": info.major_parties if info else "",
            "key_leaders": info.key_leaders if info else "",
            "result_2019": info.result_2019 if info else "",
            "notes": info.notes if info else "",
            "tweet_mentions": total,
            "labeled_tweets": labeled_n,
            "positive": int(counts.get("positive", 0)),
            "neutral": int(counts.get("neutral", 0)),
            "negative": int(counts.get("negative", 0)),
            "positive_pct": round(counts.get("positive", 0) / labeled_n, 4) if labeled_n else 0,
            "neutral_pct": round(counts.get("neutral", 0) / labeled_n, 4) if labeled_n else 0,
            "negative_pct": round(counts.get("negative", 0) / labeled_n, 4) if labeled_n else 0,
        }
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading tweets...")
    df = load_all_tweets()
    print(f"  {len(df):,} unique tweets")

    tagged = build_state_rows(df)
    print(f"  {len(tagged):,} tweets matched at least one state")

    exploded = explode_by_state(tagged)
    exploded.to_csv(RESULTS_DIR / "state_tweets.csv", index=False)

    summary = summarize_states(exploded)
    (RESULTS_DIR / "state_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    top = sorted(
        summary["states"].items(),
        key=lambda x: x[1]["tweet_mentions"],
        reverse=True,
    )[:10]
    print("\nTop 10 states by tweet mentions:")
    for name, data in top:
        print(f"  {name}: {data['tweet_mentions']:,} mentions")

    print(f"\nSaved {RESULTS_DIR / 'state_tweets.csv'}")
    print(f"Saved {RESULTS_DIR / 'state_summary.json'}")


if __name__ == "__main__":
    main()
