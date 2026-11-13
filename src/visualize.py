from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from wordcloud import WordCloud

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#c0392b",
    "neutral": "#7f8c8d",
    "positive": "#1e8449",
}
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pie(counts: pd.Series, path: Path, title: str) -> None:
    values = [int(counts.get(label, 0)) for label in SENTIMENT_ORDER]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(
        values,
        labels=SENTIMENT_ORDER,
        colors=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title(title)
    _save(fig, path)


def plot_bar(counts: pd.Series, path: Path, title: str, ylabel: str) -> None:
    values = [int(counts.get(label, 0)) for label in SENTIMENT_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(SENTIMENT_ORDER, values, color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel(ylabel)
    ax.bar_label(bars, fmt="%d")
    _save(fig, path)


def plot_model_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    plot_df = metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on held-out test tweets")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score (weighted)")
    ax.legend(title="Metric", loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    _save(fig, path)


def plot_confusion(y_true, y_pred, path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=SENTIMENT_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(matrix, display_labels=SENTIMENT_ORDER).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    _save(fig, path)


def plot_trend(labels: pd.Series, path: Path, window: int = 4000) -> None:
    encoded = pd.get_dummies(labels).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    rolled = encoded.rolling(window, min_periods=max(200, window // 10)).mean()
    x = np.arange(len(rolled))
    fig, ax = plt.subplots(figsize=(10, 5))
    for sentiment in SENTIMENT_ORDER:
        ax.plot(x, rolled[sentiment], label=sentiment, color=SENTIMENT_COLORS[sentiment], linewidth=2)
    ax.set_title(f"Sentiment share over tweet collection order (rolling window = {window})")
    ax.set_xlabel("Tweet index in 2019 election corpus")
    ax.set_ylabel("Share of tweets")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    _save(fig, path)


def plot_leader_sentiment(leader_df: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(leader_df["leader"], leader_df["pred"], normalize="index")
    table = table.reindex(columns=SENTIMENT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    table.plot(kind="bar", ax=ax, color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER], rot=0)
    ax.set_title("Predicted sentiment share by political figure")
    ax.set_xlabel("Political figure mentioned in tweet")
    ax.set_ylabel("Share of tweets")
    ax.set_ylim(0, 1)
    ax.legend(title="Sentiment")
    _save(fig, path)


def plot_wordclouds(df: pd.DataFrame, text_col: str, label_col: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    font = str(FONT_PATH) if FONT_PATH.exists() else None
    for ax, sentiment in zip(axes, SENTIMENT_ORDER):
        text = " ".join(df.loc[df[label_col] == sentiment, text_col].astype(str).tolist())
        if len(text.split()) < 5:
            ax.text(0.5, 0.5, f"Too few {sentiment} tokens", ha="center", va="center")
            ax.axis("off")
            continue
        cloud = WordCloud(
            width=700,
            height=500,
            background_color="white",
            font_path=font,
            colormap="viridis" if sentiment == "positive" else ("Reds" if sentiment == "negative" else "cividis"),
            max_words=80,
        ).generate(text)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(sentiment.capitalize())
        ax.axis("off")
    fig.suptitle("Most frequent terms by sentiment class", fontsize=13)
    _save(fig, path)
