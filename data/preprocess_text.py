"""
preprocess_text.py
──────────────────
Preprocesses the Jigsaw Toxic Comment Classification dataset.

Expected input:  data/raw/train.csv   (from Jigsaw Kaggle competition)
Outputs:
  - data/processed/text_splits.json      (train/val/test clean text lists)
  - data/processed/text_y_train.npy      (6 binary labels)
  - data/processed/text_y_val.npy
  - data/processed/text_y_test.npy
  - data/processed/text_pos_weights.npy  (for weighted BCE loss)
  - data/processed/raw_text_val.npy      (original strings for dashboard)

Run:
    python data/preprocess_text.py
"""

import os
import re
import json
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# ── Download NLTK data ────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH = os.path.join(os.path.dirname(__file__), "raw", "train.csv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
VAL_SIZE      = 0.10
TEST_SIZE     = 0.10

LABEL_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate",
]

# ── Clean text ────────────────────────────────────────────────────────────────
_stop_words  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip noise, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"<.*?>", " ", text)                    # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # non-alpha
    text = re.sub(r"\s+", " ", text).strip()              # extra spaces

    tokens = text.split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 1
    ]
    return " ".join(tokens)


# ── Tokenize with DistilBERT ──────────────────────────────────────────────────

def tokenize_batch(texts: list, tokenizer) -> np.ndarray:
    """
    Returns input_ids as int32 array shaped [N, MAX_LEN].
    DistilBERT only needs input_ids (no token_type_ids).
    Attention masks are stored separately but we save both.
    """
    enc = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_pos_weights(y: np.ndarray) -> np.ndarray:
    """
    Per-label positive class weight = (neg_count / pos_count).
    Passed directly to tf.nn.weighted_cross_entropy_with_logits.
    """
    pos = y.sum(axis=0)
    neg = len(y) - pos
    weights = neg / (pos + 1e-6)
    print("[INFO] Positive class weights per label:")
    for label, w in zip(LABEL_COLS, weights):
        print(f"       {label}: {w:.2f}")
    return weights.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Loading raw text data from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"[INFO] Raw shape: {df.shape}")

    # Fill missing comments
    df["comment_text"] = df["comment_text"].fillna("")

    # ── Clean ─────────────────────────────────────────────────────────────────
    print("[INFO] Cleaning text (this may take a minute)...")
    df["clean_text"] = df["comment_text"].apply(clean_text)

    # Drop rows with empty cleaned text
    df = df[df["clean_text"].str.strip() != ""]
    print(f"[INFO] After cleaning: {len(df):,} rows")

    # ── Labels ────────────────────────────────────────────────────────────────
    y = df[LABEL_COLS].values.astype(np.float32)
    texts = df["clean_text"].tolist()
    raw_texts = df["comment_text"].tolist()   # original for dashboard

    dist = y.sum(axis=0)
    print("[INFO] Label distribution:")
    for label, cnt in zip(LABEL_COLS, dist):
        print(f"       {label}: {int(cnt):,}  ({cnt/len(y)*100:.1f}%)")

    # ── Train / Val / Test split ───────────────────────────────────────────────
    # Stratify on 'toxic' (most common label) for reproducibility
    y_strat = y[:, 0].astype(int)
    idx = np.arange(len(texts))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=VAL_SIZE + TEST_SIZE, random_state=RANDOM_STATE, stratify=y_strat
    )
    y_strat_tmp = y_strat[idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_strat_tmp
    )

    texts_train = [texts[i] for i in idx_train]
    texts_val   = [texts[i] for i in idx_val]
    texts_test  = [texts[i] for i in idx_test]
    raw_val     = [raw_texts[i] for i in idx_val]

    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_test  = y[idx_test]

    print(f"[INFO] Split → Train: {len(texts_train):,}, Val: {len(texts_val):,}, "
          f"Test: {len(texts_test):,}")

    # ── Positive class weights ────────────────────────────────────────────────
    pos_weights = compute_pos_weights(y_train)

    # ── Save ─────────────────────────────────────────────────────────────────
    # Save text splits as JSON (keras_hub handles tokenization internally)
    splits = {"train": texts_train, "val": texts_val, "test": texts_test}
    with open(os.path.join(OUT_DIR, "text_splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)

    np.save(os.path.join(OUT_DIR, "text_y_train.npy"),      y_train)
    np.save(os.path.join(OUT_DIR, "text_y_val.npy"),        y_val)
    np.save(os.path.join(OUT_DIR, "text_y_test.npy"),       y_test)
    np.save(os.path.join(OUT_DIR, "text_pos_weights.npy"),  pos_weights)

    # Save raw val comments for dashboard display
    np.save(os.path.join(OUT_DIR, "raw_text_val.npy"),
            np.array(raw_val, dtype=object))

    print("[DONE] Text preprocessing complete. Files saved to data/processed/")


if __name__ == "__main__":
    main()
