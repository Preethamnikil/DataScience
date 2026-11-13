"""End-to-end sentiment analysis pipeline for 2019 Indian election tweets."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_tweet, map_label
from src.visualize import (
    plot_bar,
    plot_confusion,
    plot_leader_sentiment,
    plot_model_comparison,
    plot_pie,
    plot_trend,
    plot_wordclouds,
)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}

LSTM_TRAIN_SIZE = 16000
TRANSFORMER_TRAIN_SIZE = 10000
MAX_VOCAB = 8000
MAX_LEN = 40
RANDOM_STATE = 42


def timed(name: str, fn, *args, **kwargs):
    start = time.perf_counter()
    print(f"\n[{name}] started")
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[{name}] done in {elapsed:.1f}s")
    return result, elapsed


def load_election_tweets() -> pd.DataFrame:
    labeled = pd.read_csv(DATA_DIR / "Twitter_Data.csv")
    labeled = labeled.dropna(subset=["clean_text", "category"]).copy()
    labeled["text"] = labeled["clean_text"].astype(str)
    labeled["label"] = labeled["category"].map(map_label)
    labeled = labeled.dropna(subset=["label"])
    labeled["source"] = "Twitter_Data"
    labeled["leader"] = np.where(
        labeled["text"].str.contains(r"\brahul\b", case=False, regex=True),
        "Rahul",
        np.where(labeled["text"].str.contains(r"\bmodi\b", case=False, regex=True), "Modi", "Other"),
    )
    return labeled[["text", "label", "leader", "source"]].reset_index(drop=True)


def load_leader_raw_tweets() -> pd.DataFrame:
    frames = []
    for name, leader in (("Modi.csv", "Modi"), ("Rahul.csv", "Rahul")):
        raw = pd.read_csv(DATA_DIR / name)
        frame = pd.DataFrame({"text": raw["Tweet"].astype(str), "leader": leader})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score(y_true, y_pred, model_name: str, seconds: float) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "train_seconds": round(seconds, 1),
    }


def train_ml_models(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    specs = [
        ("Naive Bayes", MultinomialNB()),
        ("SVM", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    rows = []
    reports = []
    predictions = {}
    best_name, best_f1, best_model = None, -1, None

    for name, model in specs:
        start = time.perf_counter()
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        elapsed = time.perf_counter() - start
        row = score(y_test, pred, name, elapsed)
        rows.append(row)
        predictions[name] = pred
        reports.append(f"\n===== {name} =====\n{classification_report(y_test, pred, digits=4)}")
        print(f"  {name}: acc={row['accuracy']:.4f} f1={row['f1']:.4f}")
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        if row["f1"] > best_f1:
            best_name, best_f1, best_model = name, row["f1"], model

    return {
        "rows": rows,
        "reports": reports,
        "predictions": predictions,
        "vectorizer": vectorizer,
        "best_name": best_name,
        "best_model": best_model,
    }


def _prepare_sequences(train_texts, test_texts):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<unk>")
    tokenizer.fit_on_texts(train_texts)
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN)
    return tokenizer, x_train, x_test


def _to_ids(labels: pd.Series) -> np.ndarray:
    return np.array([LABEL_TO_ID[label] for label in labels], dtype=np.int32)


def _class_weights(y_ids: np.ndarray) -> dict:
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_ids)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def train_lstm(train_texts, test_texts, y_train, y_test):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    if len(train_texts) > LSTM_TRAIN_SIZE:
        idx = rng.choice(len(train_texts), LSTM_TRAIN_SIZE, replace=False)
        train_texts = train_texts.iloc[idx]
        y_train = y_train.iloc[idx]

    tokenizer, x_train, x_test = _prepare_sequences(train_texts, test_texts)
    y_train_ids = _to_ids(y_train)
    y_test_ids = _to_ids(y_test)

    model = keras.Sequential(
        [
            layers.Input(shape=(MAX_LEN,)),
            layers.Embedding(MAX_VOCAB, 128),
            layers.SpatialDropout1D(0.2),
            layers.LSTM(64, dropout=0.25),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.perf_counter()
    model.fit(
        x_train,
        y_train_ids,
        validation_split=0.1,
        epochs=3,
        batch_size=128,
        class_weight=_class_weights(y_train_ids),
        verbose=2,
    )
    elapsed = time.perf_counter() - start
    pred_ids = model.predict(x_test, batch_size=256, verbose=0).argmax(axis=1)
    pred = pd.Series(pred_ids).map({idx: label for label, idx in LABEL_TO_ID.items()}).tolist()
    model.save(MODELS_DIR / "lstm.keras")
    joblib.dump(tokenizer, MODELS_DIR / "lstm_tokenizer.joblib")
    row = score(y_test, pred, "LSTM", elapsed)
    report = f"\n===== LSTM =====\n{classification_report(y_test, pred, digits=4)}"
    print(f"  LSTM: acc={row['accuracy']:.4f} f1={row['f1']:.4f}")
    return row, report, pred


def train_transformer(train_texts, test_texts, y_train, y_test):
    """Compact BERT-style transformer encoder (self-attention + FFN)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    if len(train_texts) > TRANSFORMER_TRAIN_SIZE:
        idx = rng.choice(len(train_texts), TRANSFORMER_TRAIN_SIZE, replace=False)
        train_texts = train_texts.iloc[idx]
        y_train = y_train.iloc[idx]

    tokenizer, x_train, x_test = _prepare_sequences(train_texts, test_texts)
    y_train_ids = _to_ids(y_train)
    y_test_ids = _to_ids(y_test)

    embed_dim = 64
    num_heads = 4
    ff_dim = 128

    inputs = layers.Input(shape=(MAX_LEN,))
    x = layers.Embedding(MAX_VOCAB, embed_dim)(inputs)
    positions = tf.range(start=0, limit=MAX_LEN, delta=1)
    x = x + layers.Embedding(MAX_LEN, embed_dim)(positions)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(0.1)(attn))
    ffn = layers.Dense(embed_dim)(layers.Dense(ff_dim, activation="relu")(x))
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(0.1)(ffn))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="bert_style_transformer")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    start = time.perf_counter()
    model.fit(
        x_train,
        y_train_ids,
        validation_split=0.1,
        epochs=3,
        batch_size=128,
        class_weight=_class_weights(y_train_ids),
        verbose=2,
    )
    elapsed = time.perf_counter() - start
    pred_ids = model.predict(x_test, batch_size=256, verbose=0).argmax(axis=1)
    pred = pd.Series(pred_ids).map({idx: label for label, idx in LABEL_TO_ID.items()}).tolist()
    model.save(MODELS_DIR / "bert_style_transformer.keras")
    row = score(y_test, pred, "BERT-style Transformer", elapsed)
    report = (
        f"\n===== BERT-style Transformer =====\n"
        f"{classification_report(y_test, pred, digits=4)}"
    )
    print(f"  BERT-style Transformer: acc={row['accuracy']:.4f} f1={row['f1']:.4f}")
    return row, report, pred


def predict_examples(vectorizer, model) -> list[dict]:
    samples = [
        "Modi government is doing a great job for development and national security.",
        "Rahul Gandhi has no vision and this campaign is a complete disaster.",
        "Voting will take place on 23rd May across polling booths in India.",
        "Chowkidar chor hai, people are tired of empty promises and unemployment.",
        "Proud of our democracy, both parties should debate jobs and agriculture calmly.",
    ]
    cleaned = [clean_tweet(text) for text in samples]
    preds = model.predict(vectorizer.transform(cleaned))
    return [
        {"tweet": text, "clean_text": clean, "predicted_sentiment": pred}
        for text, clean, pred in zip(samples, cleaned, preds)
    ]


def main() -> None:
    warnings.filterwarnings("ignore")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    labeled, load_s = timed("load labeled tweets", load_election_tweets)
    print(labeled["label"].value_counts().to_string())
    print("leader mix\n", labeled["leader"].value_counts().to_string())

    labeled["clean_text"], prep_s = timed("preprocess", lambda: labeled["text"].map(clean_tweet))
    labeled = labeled[labeled["clean_text"].str.split().str.len() >= 3].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        labeled["clean_text"],
        labeled["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labeled["label"],
    )

    ml, ml_s = timed("train traditional ML", train_ml_models, X_train, X_test, y_train, y_test)
    lstm_row, lstm_report, lstm_pred = timed(
        "train LSTM", train_lstm, X_train, X_test, y_train, y_test
    )[0]
    bert_row, bert_report, bert_pred = timed(
        "train BERT-style Transformer",
        train_transformer,
        X_train,
        X_test,
        y_train,
        y_test,
    )[0]

    metrics = pd.DataFrame(ml["rows"] + [lstm_row, bert_row])
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    reports = "".join(ml["reports"] + [lstm_report, bert_report])
    (RESULTS_DIR / "classification_reports.txt").write_text(reports, encoding="utf-8")

    best_ml_name = ml["best_name"]
    best_ml_pred = ml["predictions"][best_ml_name]
    best_overall = metrics.sort_values("f1", ascending=False).iloc[0]
    pred_lookup = {
        **ml["predictions"],
        "LSTM": lstm_pred,
        "BERT-style Transformer": bert_pred,
    }
    overall_pred = pred_lookup[best_overall["model"]]

    plot_pie(
        labeled["label"].value_counts(),
        FIGURES_DIR / "fig1_sentiment_pie.png",
        "Overall sentiment in 2019 Indian election tweets",
    )
    plot_bar(
        labeled["label"].value_counts(),
        FIGURES_DIR / "fig2_sentiment_bar.png",
        "Tweet counts by sentiment class",
        "Number of tweets",
    )
    plot_trend(labeled["label"], FIGURES_DIR / "fig3_sentiment_trend.png")
    plot_wordclouds(
        labeled,
        "clean_text",
        "label",
        FIGURES_DIR / "fig4_wordclouds.png",
    )
    plot_model_comparison(metrics, FIGURES_DIR / "fig5_model_comparison.png")
    plot_confusion(
        y_test,
        overall_pred,
        FIGURES_DIR / "fig6_confusion_matrix.png",
        f"Confusion matrix — {best_overall['model']}",
    )

    leader_raw = load_leader_raw_tweets()
    leader_raw["clean_text"] = leader_raw["text"].map(clean_tweet)
    leader_raw = leader_raw[leader_raw["clean_text"].str.split().str.len() >= 3].copy()
    leader_raw["pred"] = ml["best_model"].predict(
        ml["vectorizer"].transform(leader_raw["clean_text"])
    )
    plot_leader_sentiment(leader_raw, FIGURES_DIR / "fig7_leader_sentiment.png")
    plot_pie(
        leader_raw.loc[leader_raw["leader"] == "Modi", "pred"].value_counts(),
        FIGURES_DIR / "fig8_modi_pie.png",
        "Predicted sentiment — tweets mentioning Modi",
    )
    plot_pie(
        leader_raw.loc[leader_raw["leader"] == "Rahul", "pred"].value_counts(),
        FIGURES_DIR / "fig9_rahul_pie.png",
        "Predicted sentiment — tweets mentioning Rahul Gandhi",
    )

    examples = predict_examples(ml["vectorizer"], ml["best_model"])
    pd.DataFrame(examples).to_csv(RESULTS_DIR / "example_predictions.csv", index=False)

    sample = pd.DataFrame(
        {
            "raw_text": leader_raw["text"].head(12).tolist(),
            "clean_text": leader_raw["clean_text"].head(12).tolist(),
            "predicted_sentiment": leader_raw["pred"].head(12).tolist(),
        }
    )
    sample.to_csv(RESULTS_DIR / "preprocessing_samples.csv", index=False)

    summary = {
        "dataset": "2019 Indian General Election tweets (Twitter_Data.csv)",
        "topic": "Public opinion on Indian politics during Lok Sabha 2019",
        "n_labeled_after_cleaning": int(len(labeled)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split": "80/20 stratified",
        "label_counts": labeled["label"].value_counts().to_dict(),
        "leader_counts_labeled": labeled["leader"].value_counts().to_dict(),
        "leader_raw_counts": leader_raw["leader"].value_counts().to_dict(),
        "leader_sentiment": (
            pd.crosstab(leader_raw["leader"], leader_raw["pred"], normalize="index")
            .round(4)
            .to_dict()
        ),
        "metrics": metrics.to_dict(orient="records"),
        "best_ml_model": best_ml_name,
        "best_overall_model": best_overall["model"],
        "best_overall_f1": float(best_overall["f1"]),
        "example_predictions": examples,
        "timing_seconds": {"load": round(load_s, 1), "preprocess": round(prep_s, 1), "ml": round(ml_s, 1)},
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table_lines = [
        "Model | Accuracy | Precision | Recall | F1-Score",
        "---|---|---|---|---",
    ]
    for row in metrics.itertuples(index=False):
        table_lines.append(
            f"{row.model} | {row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f}"
        )
    (RESULTS_DIR / "ieee_table.md").write_text("\n".join(table_lines), encoding="utf-8")

    print("\n========== FINAL RESULTS ==========")
    print(metrics.to_string(index=False))
    print("\nBest overall model:", best_overall["model"], "F1=", best_overall["f1"])
    print("Figures saved to", FIGURES_DIR)
    print("Metrics saved to", RESULTS_DIR / "metrics.csv")


if __name__ == "__main__":
    main()
