"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        PUBG TOXICITY DETECTOR — GOOGLE COLAB TRAINING SCRIPT               ║
║  Self-contained: Preprocessing → DistilBERT Fine-tuning → Evaluation       ║
║                                                                              ║
║  SETUP:                                                                      ║
║    1. Runtime → Change runtime type → GPU (T4 recommended)                  ║
║    2. Upload train.csv via the sidebar Files panel (or drag-and-drop)        ║
║       The file will be at /content/train.csv                                 ║
║    3. Run this script — it mounts Drive and saves results there             ║
║                                                                              ║
║  GPU:  Recommended (T4 free tier is sufficient)                              ║
║  RAM:  ~12 GB (Colab standard)                                               ║
║  Time: ~35–50 min on T4 GPU                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
#  SECTION 0 — GPU CHECK (must run first before heavy imports)
# ═══════════════════════════════════════════════════════════════════
import subprocess, sys, os
import tensorflow as tf

# ── Configure GPU memory growth (prevents OOM on Colab T4) ────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU detected  : {tf.test.gpu_device_name()}")
    print(f"  Physical GPUs : {gpus}")
else:
    print("=" * 60)
    print("  ⚠  NO GPU RUNTIME DETECTED")
    print("  Training DistilBERT on CPU takes 50+ hours.")
    print("  → Go to:  Runtime → Change runtime type → T4 GPU")
    print("  → Then:   Runtime → Restart session and run all")
    print("=" * 60)
    raise RuntimeError(
        "GPU required. Enable it via Runtime → Change runtime type → GPU"
    )

# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — INSTALL & IMPORTS
# ═══════════════════════════════════════════════════════════════════
def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("keras-hub", "transformers", "tokenizers", "nltk")

import re, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import keras_hub
from transformers import DistilBertTokenizerFast

warnings.filterwarnings("ignore")
print(f"TensorFlow : {tf.__version__}")

# ─── Enable mixed precision for ~2x faster GPU training ───────────
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("✓ Mixed precision (float16) enabled")

# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — GOOGLE DRIVE MOUNT & PATHS
# ═══════════════════════════════════════════════════════════════════
print("\n[Setup] Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/pubg_ai"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print(f"✓ Drive mounted. Working folder: {DRIVE_DIR}")
    IN_COLAB = True
except Exception:
    # Running locally (for testing), fallback
    DRIVE_DIR = "/tmp/pubg_ai"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    IN_COLAB = False
    print("  (Not in Colab — using /tmp/pubg_ai as fallback)")

OUT_DIR = os.path.join(DRIVE_DIR, "saved_models")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"✓ Output will be saved to: {OUT_DIR}")

RANDOM_STATE  = 42
LABEL_COLS    = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DISTILBERT_ID = "distilbert-base-uncased"
KERA_PRESET   = "distil_bert_base_en_uncased"
MAX_LEN       = 128
BATCH_SIZE    = 32
EPOCHS        = 4

# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — DATASET PATH (uploaded directly to Colab)
# ═══════════════════════════════════════════════════════════════════
# The user uploaded train.csv via the Colab Files sidebar.
# Default upload location is /content/train.csv
DATA_PATH = "/content/train.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Please upload train.csv using the Colab Files panel (left sidebar → upload icon)"
    )
print(f"\n[Dataset] Found data at: {DATA_PATH}")

# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print(f"\n[1] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(df.head(3))

# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — EDA
# ═══════════════════════════════════════════════════════════════════
print("\n[2] EDA")
label_counts = df[LABEL_COLS].sum().sort_values(ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(label_counts.index, label_counts.values, color="#4c6ef5")
axes[0].set_title("Label Frequency"); axes[0].set_xlabel("Count")

multi_label_count = (df[LABEL_COLS].sum(axis=1) > 1).sum()
axes[1].pie(
    [multi_label_count, len(df) - multi_label_count],
    labels=["Multi-label", "Single/None"],
    autopct="%1.1f%%",
    colors=["#4c6ef5", "#ff6b6b"]
)
axes[1].set_title("Multi-label Distribution")
plt.suptitle("Jigsaw Toxic Comment Dataset — EDA", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_eda.png"), dpi=120, bbox_inches="tight")
plt.close()
print("    EDA plot saved")

# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 — TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Text Preprocessing...")

_stop  = set(stopwords.words("english"))
_lemma = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(_lemma.lemmatize(t) for t in text.split() if t not in _stop and len(t) > 1)

df["comment_text"] = df["comment_text"].fillna("")
df["clean_text"]   = df["comment_text"].apply(clean_text)
df = df[df["clean_text"].str.strip() != ""]
print(f"    After cleaning: {len(df):,} rows")

y       = df[LABEL_COLS].values.astype(np.float32)
texts   = df["clean_text"].tolist()
y_strat = y[:, 0].astype(int)
idx     = np.arange(len(texts))

idx_train, idx_tmp = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_strat)
idx_val, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_strat[idx_tmp])

texts_train = [texts[i] for i in idx_train]
texts_val   = [texts[i] for i in idx_val]
texts_test  = [texts[i] for i in idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

print(f"    Train: {len(texts_train):,}  Val: {len(texts_val):,}  Test: {len(texts_test):,}")

pos         = y_train.sum(axis=0)
neg         = len(y_train) - pos
pos_weights = (neg / (pos + 1e-6)).astype(np.float32)
print("    Class weights:", dict(zip(LABEL_COLS, pos_weights.round(1))))

# ─── Tokenization ─────────────────────────────────────────────────
print(f"\n    Loading tokenizer: {DISTILBERT_ID}")
tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_ID)

def tokenize(texts_list):
    enc = tokenizer(texts_list, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)

print("    Tokenizing train..."); X_train_ids, X_train_mask = tokenize(texts_train)
print("    Tokenizing val...");   X_val_ids,   X_val_mask   = tokenize(texts_val)
print("    Tokenizing test...");  X_test_ids,  X_test_mask  = tokenize(texts_test)
print(f"    Token shapes — Train: {X_train_ids.shape}")

# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 — MODEL BUILD
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Building DistilBERT Model (keras_hub backbone)")

def build_distilbert(max_len=128, num_labels=6):
    backbone = keras_hub.models.DistilBertBackbone.from_preset(KERA_PRESET)
    backbone.trainable = True
    # Freeze all but last 2 transformer layers (fine-tuning only top layers)
    for layer in backbone.layers[:-2]:
        layer.trainable = False

    inp_ids  = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    inp_mask = keras.Input(shape=(max_len,), dtype="int32", name="padding_mask")
    seq_out  = backbone({"token_ids": inp_ids, "padding_mask": inp_mask})
    cls_out  = seq_out[:, 0, :]             # [CLS] token representation
    x   = layers.Dropout(0.3)(cls_out)
    x   = layers.Dense(256, activation="gelu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(num_labels, dtype="float32", activation="sigmoid",
                       name="toxicity_scores")(x)
    return keras.Model(inputs=[inp_ids, inp_mask], outputs=out,
                       name="ToxicityDetector_DistilBERT")

def weighted_bce_loss(pos_weights_arr):
    pw = tf.constant(pos_weights_arr, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        bce    = -(pw * y_true * tf.math.log(y_pred) +
                   (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)
    return loss

class MacroAUC(keras.metrics.Metric):
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

# ─── Build tf.data datasets ───────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════
#  SECTION 7 — TRAINING
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Training DistilBERT Toxicity Detector")
print(f"    Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Max Len: {MAX_LEN}")

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

# ─── Training curves ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history.history["loss"],              label="Train", color="#4c6ef5")
axes[0].plot(history.history.get("val_loss", []),  label="Val",   color="#ff6b6b")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(history.history.get("macro_auc", []),     label="Train AUC", color="#4c6ef5")
axes[1].plot(history.history.get("val_macro_auc", []), label="Val AUC",   color="#ff6b6b")
axes[1].set_title("Macro AUC"); axes[1].legend()
plt.suptitle("DistilBERT Toxicity Detector — Training History", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_history.png"), dpi=120, bbox_inches="tight")
plt.close()

# ─── Save model ───────────────────────────────────────────────────
tox_save_path = os.path.join(OUT_DIR, "toxicity_detector_tf.keras")
model.save(tox_save_path)
print(f"    Model saved → {tox_save_path}")

# ═══════════════════════════════════════════════════════════════════
#  SECTION 8 — EVALUATION
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Evaluation on Test Set")

THRESHOLD  = 0.5
y_pred_raw = model.predict(test_ds, verbose=1)
y_pred_bin = (y_pred_raw >= THRESHOLD).astype(int)

print("\n── Per-Label Classification Report ──────────────────────")
print(classification_report(y_test, y_pred_bin, target_names=LABEL_COLS, zero_division=0))

per_auc, per_f1 = {}, {}
for i, label in enumerate(LABEL_COLS):
    if y_test[:, i].sum() > 0:
        per_auc[label] = round(float(roc_auc_score(y_test[:, i], y_pred_raw[:, i])), 4)
        per_f1[label]  = round(float(f1_score(y_test[:, i], y_pred_bin[:, i], zero_division=0)), 4)

macro_auc = float(np.mean(list(per_auc.values())))
macro_f1  = float(np.mean(list(per_f1.values())))
print(f"\n  Macro AUC : {macro_auc:.4f}")
print(f"  Macro F1  : {macro_f1:.4f}")

# ─── Per-label metric bar charts ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
auc_vals   = list(per_auc.values())
f1_vals    = list(per_f1.values())
auc_colors = ["#2ea043" if v >= 0.90 else "#e3b341" if v >= 0.70 else "#da3633"
               for v in auc_vals]

axes[0].barh(list(per_auc.keys()), auc_vals, color=auc_colors)
axes[0].axvline(x=0.90, color="#da3633", linestyle="--", alpha=0.6, label="Target (0.90)")
axes[0].set_xlim(0, 1); axes[0].set_title("Per-Label AUC"); axes[0].legend()
axes[1].barh(list(per_f1.keys()), f1_vals, color="#388bfd")
axes[1].set_xlim(0, 1); axes[1].set_title("Per-Label F1")
plt.suptitle("Toxicity Detector — Evaluation Metrics", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toxicity_metrics_plot.png"), dpi=120, bbox_inches="tight")
plt.close()

# ─── Qualitative examples ─────────────────────────────────────────
print("\n[7] Qualitative Examples")
sample_comments = [
    "Watch out, they are flanking from the east!",
    "I will destroy you, you worthless noob.",
    "Great teamwork! Let's push the final circle.",
    "You are so stupid, why did you go there??",
    "Cover me while I revive Jake.",
    "Get out of this game you disgusting idiot.",
    "Nice shot! That was insane!",
    "I know where you live, I'll find you.",
]
clean_samples = [clean_text(c) for c in sample_comments]
enc_s         = tokenizer(clean_samples, max_length=MAX_LEN, padding="max_length",
                           truncation=True, return_tensors="np")
sample_preds  = model.predict(
    {"token_ids":    enc_s["input_ids"].astype(np.int32),
     "padding_mask": enc_s["attention_mask"].astype(np.int32)},
    verbose=0
)

print(f"\n  {'Comment':<50}  {'Toxic':>6}  {'Obscene':>8}  {'Threat':>8}  {'Insult':>8}")
print("  " + "─" * 82)
for comment, pred in zip(sample_comments, sample_preds):
    print(f"  {comment[:48]:<50}  {pred[0]:>6.3f}  {pred[2]:>8.3f}  {pred[3]:>8.3f}  {pred[4]:>8.3f}")

# ─── Save metrics JSON ────────────────────────────────────────────
metrics = {
    "macro_auc":     round(macro_auc, 4),
    "macro_f1":      round(macro_f1,  4),
    "per_label_auc": per_auc,
    "per_label_f1":  per_f1,
    "threshold":     THRESHOLD,
}
with open(os.path.join(OUT_DIR, "toxicity_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'═' * 55}")
print("  TOXICITY DETECTOR TRAINING COMPLETE")
print(f"  Macro AUC : {macro_auc:.4f}")
print(f"  Macro F1  : {macro_f1:.4f}")
print(f"{'═' * 55}")
print(f"\n  Files saved to Google Drive → {OUT_DIR}/")
print("  - toxicity_detector_tf.keras   (Main model)")
print("  - toxicity_best.keras          (Best checkpoint)")
print("  - toxicity_metrics.json        (Metrics)")
print("  - toxicity_*.png               (Plots)")
print("\n  ✓ Download these from Drive to E:\\ML\\pubg_ai\\saved_models\\")
