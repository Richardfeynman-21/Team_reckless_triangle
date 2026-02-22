"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        PUBG TOXICITY DETECTOR — KAGGLE TRAINING SCRIPT                     ║
║  Self-contained: Preprocessing → DistilBERT Fine-tuning → Evaluation       ║
║                                                                              ║
║  KAGGLE DATASET PATHS (add to your notebook before running):                ║
║    /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv    ║
║                                                                              ║
║  GPU:  REQUIRED for reasonable training time (T4 or P100)                   ║
║  RAM:  ~8 GB                                                                 ║
║  Time: ~35–50 min on Kaggle T4 GPU                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
#  SECTION 0 — INSTALL & IMPORTS
# ═══════════════════════════════════════════════════════════════════
import subprocess, sys

# Install any missing packages
def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("transformers", "tokenizers", "nltk")

import os, re, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

import keras_hub                                  # replaces removed TFAutoModel
from transformers import DistilBertTokenizerFast   # for tokenization on Windows

warnings.filterwarnings("ignore")
print(f"TensorFlow : {tf.__version__}")
print(f"GPUs       : {tf.config.list_physical_devices('GPU')}")

# ─── Enable mixed precision for faster GPU training ───────────────
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision (float16) enabled")

# ─── Output directory ───────────────────────────────────────────
if os.path.exists("/kaggle"):
    OUT_DIR = "/kaggle/working"
else:
    OUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath("."))),
        "pubg_ai", "saved_models"
    )
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE  = 42
LABEL_COLS    = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DISTILBERT_ID = "distilbert-base-uncased"
KERA_PRESET   = "distil_bert_base_en_uncased"   # keras_hub preset
MAX_LEN       = 128


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════
KAGGLE_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
# Local path: data is in the archive folder
LOCAL_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath("."))),
    "pubg_ai", "archive", "train.csv"
)
DATA_PATH   = KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else LOCAL_PATH

print(f"\n[1] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(df.head(3))


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — EDA
# ═══════════════════════════════════════════════════════════════════
print("\n[2] EDA")

# Label distribution
label_counts = df[LABEL_COLS].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(label_counts.index, label_counts.values, color="#4c6ef5")
ax.bar_label(bars, [f"{v:,}" for v in label_counts.values], padding=5)
ax.set_title("Jigsaw — Label Frequency")
ax.set_xlabel("Count (out of 159,571 comments)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_label_dist.png"), dpi=120, bbox_inches="tight")
plt.show()

# Comment length distribution
df["comment_len"] = df["comment_text"].fillna("").apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df["comment_len"].clip(upper=300), bins=60, color="#4c6ef5", edgecolor="none")
ax.axvline(x=MAX_LEN, color="#da3633", linestyle="--", label=f"MAX_LEN={MAX_LEN}")
ax.set_title("Comment Length Distribution (words)")
ax.set_xlabel("Word count")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_comment_len.png"), dpi=120, bbox_inches="tight")
plt.show()

# Label correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[LABEL_COLS].corr(), annot=True, fmt=".2f",
            cmap="YlOrRd", linewidths=0.5, ax=ax)
ax.set_title("Label Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_label_corr.png"), dpi=120, bbox_inches="tight")
plt.show()

print("    Note: 'obscene' and 'insult' are highly correlated (~0.74)")
print("    Note: Only ~9.6% of comments are labeled toxic")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Text Preprocessing (cleaning + tokenization)")

_stop   = set(stopwords.words("english"))
_lemma  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"<.*?>", " ", text)                    # HTML
    text = re.sub(r"[^a-z\s]", " ", text)                 # non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [_lemma.lemmatize(t) for t in text.split()
              if t not in _stop and len(t) > 1]
    return " ".join(tokens)

print("    Cleaning text (NLTK lemmatization + stopword removal)...")
df["comment_text"] = df["comment_text"].fillna("")
df["clean_text"]   = df["comment_text"].apply(clean_text)

# Drop empty rows
df = df[df["clean_text"].str.strip() != ""]
print(f"    After cleaning: {len(df):,} rows")

# ── Train/Val/Test split ───────────────────────────────────────────
y = df[LABEL_COLS].values.astype(np.float32)
texts     = df["clean_text"].tolist()
raw_texts = df["comment_text"].tolist()

y_strat = y[:, 0].astype(int)

idx = np.arange(len(texts))
idx_train, idx_tmp = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_strat
)
y_strat_tmp = y_strat[idx_tmp]
idx_val, idx_test = train_test_split(
    idx_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_strat_tmp
)

texts_train  = [texts[i] for i in idx_train]
texts_val    = [texts[i] for i in idx_val]
texts_test   = [texts[i] for i in idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

print(f"    Train: {len(texts_train):,}  Val: {len(texts_val):,}  Test: {len(texts_test):,}")

# ── Positive class weights ─────────────────────────────────────────
pos  = y_train.sum(axis=0)
neg  = len(y_train) - pos
pos_weights = (neg / (pos + 1e-6)).astype(np.float32)
print("\n    Positive class weights:")
for label, w in zip(LABEL_COLS, pos_weights):
    print(f"      {label:<18}: {w:.1f}")

# ── DistilBERT Tokenization ───────────────────────────────────────
print(f"\n    Loading DistilBERT tokenizer: {DISTILBERT_ID}")
tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_ID)

def tokenize(texts_list):
    enc = tokenizer(
        texts_list,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)

print("    Tokenizing train split...")
X_train_ids, X_train_mask = tokenize(texts_train)
print("    Tokenizing val split...")
X_val_ids,   X_val_mask   = tokenize(texts_val)
print("    Tokenizing test split...")
X_test_ids,  X_test_mask  = tokenize(texts_test)

print(f"\n    Token ID matrix shape → Train: {X_train_ids.shape}, Val: {X_val_ids.shape}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — BUILD DISTILBERT MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Building DistilBERT Model")

def build_distilbert(max_len=128, num_labels=6):
    """
    Architecture (keras_hub):
      token_ids + padding_mask (int32)
        -> DistilBertBackbone (last 2 blocks fine-tuned)
        -> [CLS] token (dim 768)
        -> Dropout(0.3) -> Dense(256, gelu) -> Dropout(0.2)
        -> Dense(6, sigmoid)  <- multi-label output
    """
    backbone = keras_hub.models.DistilBertBackbone.from_preset(KERA_PRESET)

    # Freeze all but unfreeze last 2 transformer layers
    backbone.trainable = True
    for layer in backbone.layers[:-2]:
        layer.trainable = False

    inp_ids  = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    inp_mask = keras.Input(shape=(max_len,), dtype="int32", name="padding_mask")

    seq_out  = backbone({"token_ids": inp_ids, "padding_mask": inp_mask})
    cls_out  = seq_out[:, 0, :]   # [CLS] token

    x   = layers.Dropout(0.3)(cls_out)
    x   = layers.Dense(256, activation="gelu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(num_labels, dtype="float32", activation="sigmoid",
                       name="toxicity_scores")(x)

    model = keras.Model(inputs=[inp_ids, inp_mask], outputs=out,
                        name="ToxicityDetector_DistilBERT")
    return model


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 — CUSTOM LOSS + METRIC
# ═══════════════════════════════════════════════════════════════════

def weighted_bce_loss(pos_weights_arr):
    """Per-label weighted binary cross-entropy to handle class imbalance."""
    pw = tf.constant(pos_weights_arr, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        bce = -(pw * y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)

    return loss


class MacroAUC(keras.metrics.Metric):
    """Per-label AUC averaged across all labels."""
    def __init__(self, num_labels=6, **kwargs):
        super().__init__(**kwargs)
        self._aucs = [keras.metrics.AUC(name=f"auc_{i}") for i in range(num_labels)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i, auc in enumerate(self._aucs):
            auc.update_state(y_true[:, i], y_pred[:, i])

    def result(self):
        return tf.reduce_mean([a.result() for a in self._aucs])

    def reset_state(self):
        for a in self._aucs: a.reset_state()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 — TRAINING
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Training DistilBERT Toxicity Detector")

BATCH_SIZE = 32
EPOCHS     = 4       # DistilBERT converges fast; 3–4 epochs is typical

# Build tf.data datasets using token_ids / padding_mask keys (keras_hub API)
def make_ds(ids, mask, labels, bs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        {"token_ids": ids, "padding_mask": mask}, labels
    ))
    if shuffle: ds = ds.shuffle(10_000, seed=RANDOM_STATE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train_ids, X_train_mask, y_train, BATCH_SIZE, shuffle=True)
val_ds   = make_ds(X_val_ids,   X_val_mask,   y_val,   BATCH_SIZE)
test_ds  = make_ds(X_test_ids,  X_test_mask,  y_test,  64)

model = build_distilbert()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=weighted_bce_loss(pos_weights),
    metrics=[MacroAUC(name="macro_auc")],
)

cb_list = [
    callbacks.EarlyStopping(
        monitor="val_macro_auc", patience=2,
        restore_best_weights=True, mode="max", verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(OUT_DIR, "toxicity_best.keras"),
        monitor="val_macro_auc", save_best_only=True, mode="max", verbose=0
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb_list,
)

# ── Training curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history.history["loss"],           label="Train", color="#4c6ef5")
axes[0].plot(history.history.get("val_loss",[]),label="Val",   color="#ff6b6b")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(history.history.get("macro_auc",[]),    label="Train AUC", color="#4c6ef5")
axes[1].plot(history.history.get("val_macro_auc",[]),label="Val AUC",   color="#ff6b6b")
axes[1].set_title("Macro AUC"); axes[1].legend()
plt.suptitle("DistilBERT Toxicity Detector — Training History", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_history.png"), dpi=120, bbox_inches="tight")
plt.show()

# ── Save ──────────────────────────────────────────────────────────
tox_save_path = os.path.join(OUT_DIR, "toxicity_detector_tf.keras")
model.save(tox_save_path)
print(f"    Model saved → {tox_save_path}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7 — EVALUATION
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Evaluation on Test Set")

THRESHOLD = 0.5
y_pred_raw = model.predict(test_ds, verbose=1)
y_pred_bin = (y_pred_raw >= THRESHOLD).astype(int)

print("\n── Per-Label Classification Report ─────────────────────────")
print(classification_report(y_test, y_pred_bin, target_names=LABEL_COLS, zero_division=0))

# Per-label AUC & F1
per_auc, per_f1 = {}, {}
for i, label in enumerate(LABEL_COLS):
    if y_test[:, i].sum() > 0:
        per_auc[label] = round(float(roc_auc_score(y_test[:, i], y_pred_raw[:, i])), 4)
        per_f1[label]  = round(float(f1_score(y_test[:, i], y_pred_bin[:, i], zero_division=0)), 4)

macro_auc = np.mean(list(per_auc.values()))
macro_f1  = np.mean(list(per_f1.values()))

print(f"\n  Macro AUC : {macro_auc:.4f}")
print(f"  Macro F1  : {macro_f1:.4f}")

print("\n  Per-label AUC:")
for label, auc in per_auc.items():
    bar = "█" * int(auc * 30)
    print(f"    {label:<18}: {auc:.4f}  {bar}")

# Per-label AUC bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

auc_vals = list(per_auc.values())
f1_vals  = list(per_f1.values())
auc_colors = ["#2ea043" if v >= 0.90 else "#e3b341" if v >= 0.70 else "#da3633" for v in auc_vals]

axes[0].barh(list(per_auc.keys()), auc_vals, color=auc_colors)
axes[0].axvline(x=0.90, color="#da3633", linestyle="--", alpha=0.6, label="Target (0.90)")
axes[0].set_xlim(0, 1)
axes[0].set_title("Per-Label AUC")
axes[0].legend()

axes[1].barh(list(per_f1.keys()), f1_vals, color="#388bfd")
axes[1].set_xlim(0, 1)
axes[1].set_title("Per-Label F1")

plt.suptitle("Toxicity Detector — Evaluation Metrics", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_metrics_plot.png"), dpi=120, bbox_inches="tight")
plt.show()

# ── Qualitative examples ───────────────────────────────────────────
print("\n[7] Qualitative Examples")

sample_comments = [
    "Watch out, they are flanking from the east!",
    "I will destroy you, you worthless noob.",
    "Great teamwork! Let's push the final circle.",
    "You are so stupid, why did you go there??",
    "Cover me while I revive Jake.",
    "Get out of this game you disgusting idiot.",
    "Nice shot! That was amazing!",
    "I know where you live, I'll find you.",
]

clean_samples = [clean_text(c) for c in sample_comments]
enc_s = tokenizer(clean_samples, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="np")
sample_preds = model.predict(
    {"token_ids": enc_s["input_ids"].astype(np.int32),
     "padding_mask": enc_s["attention_mask"].astype(np.int32)},
    verbose=0
)

print("\n  {'Comment':<50}  {'Toxic':>6}  {'Obscene':>8}  {'Threat':>8}  {'Insult':>8}")
print("  " + "─"*82)
for comment, pred in zip(sample_comments, sample_preds):
    row = (
        f"  {comment[:48]:<50}  "
        f"{pred[0]:>6.3f}  {pred[2]:>8.3f}  {pred[3]:>8.3f}  {pred[4]:>8.3f}"
    )
    print(row)

# Heatmap of sample predictions
fig, ax = plt.subplots(figsize=(11, 5))
short_comments = [c[:40] + "…" if len(c) > 40 else c for c in sample_comments]
sns.heatmap(
    sample_preds,
    xticklabels=LABEL_COLS,
    yticklabels=short_comments,
    annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
    linewidths=0.5, ax=ax
)
ax.set_title("Sample Prediction Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_sample_heatmap.png"), dpi=120, bbox_inches="tight")
plt.show()

# ── Save metrics JSON ──────────────────────────────────────────────
metrics = {
    "macro_auc":      round(macro_auc, 4),
    "macro_f1":       round(macro_f1,  4),
    "per_label_auc":  per_auc,
    "per_label_f1":   per_f1,
    "threshold":      THRESHOLD,
}
with open(os.path.join(OUT_DIR, "toxicity_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'═'*55}")
print("  TOXICITY DETECTOR TRAINING COMPLETE")
print(f"  Macro AUC : {macro_auc:.4f}")
print(f"  Macro F1  : {macro_f1:.4f}")
print(f"{'═'*55}")
print(f"\n  Files saved to: {OUT_DIR}/")
print("  - toxicity_detector_tf/        (TF SavedModel)")
print("  - toxicity_best.keras          (Best checkpoint)")
print("  - toxicity_metrics.json        (Metrics)")
print("  - toxicity_*.png               (Plots)")
