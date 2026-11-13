"""Classify a tweet with the trained SVM + TF-IDF pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_tweet


def predict(text: str) -> tuple[str, str]:
    vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    model = joblib.load(ROOT / "models" / "svm.joblib")
    cleaned = clean_tweet(text)
    label = model.predict(vectorizer.transform([cleaned]))[0]
    return label, cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment of an Indian-politics tweet")
    parser.add_argument("tweet", nargs="+", help="Tweet text to classify")
    args = parser.parse_args()
    text = " ".join(args.tweet)
    label, cleaned = predict(text)
    print(f"Tweet:    {text}")
    print(f"Cleaned:  {cleaned}")
    print(f"Sentiment:{label}")


if __name__ == "__main__":
    main()
