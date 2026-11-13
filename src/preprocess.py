import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "never", "nor"}
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    return LEMMATIZER.lemmatize(word)


def clean_tweet(text) -> str:
    """Tweet-specific cleaning used by every model."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub("", text)
    text = text.replace("&amp;", " and ")
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [
        _lemma(tok)
        for tok in SPACE_RE.sub(" ", text).split()
        if tok not in STOP_WORDS and (len(tok) > 2 or tok in {"no", "not"})
    ]
    return " ".join(tokens)


def map_label(value) -> str | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raw = str(value).strip().lower()
        if raw in {"positive", "pos"}:
            return "positive"
        if raw in {"negative", "neg"}:
            return "negative"
        if raw in {"neutral", "neu"}:
            return "neutral"
        return None
    if num > 0:
        return "positive"
    if num < 0:
        return "negative"
    return "neutral"
