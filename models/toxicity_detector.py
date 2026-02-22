"""
toxicity_detector.py
─────────────────────
Fine-tunes DistilBERT (via keras-hub) on the Jigsaw Toxic Comment dataset
for multi-label classification.

Labels: toxic | severe_toxic | obscene | threat | insult | identity_hate

Prerequisites:
    python data/preprocess_text.py   (must be run first)

Usage:
    python models/toxicity_detector.py              # full training
    python models/toxicity_detector.py --sample 0.1 --epochs 2   # fast test
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import keras_hub          # official Keras-native NLP; replaces TFDistilBertModel
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    multilabel_confusion_matrix,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

LABEL_COLS     = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
KERA_PRESET    = "distil_bert_base_en_uncased"   # keras_hub preset name
RANDOM_STATE   = 42


# ── Build DistilBERT Fine-tune Model (keras_hub) ─────────────────────────────

def build_distilbert_model(num_labels: int = 6, seq_len: int = 128) -> keras.Model:
    """
    Architecture (keras_hub):
      String input
        -> DistilBertPreprocessor (tokenise + pad)
        -> DistilBertBackbone     (transformer encoder)
        -> [CLS] token (dim 768)
        -> Dropout(0.3) -> Dense(256, gelu) -> Dropout(0.2)
        -> Dense(6, sigmoid)   <- multi-label output
    """
    preprocessor = keras_hub.models.DistilBertPreprocessor.from_preset(
        KERA_PRESET, sequence_length=seq_len
    )
    backbone = keras_hub.models.DistilBertBackbone.from_preset(KERA_PRESET)

    # Freeze all, unfreeze last 2 transformer layers only
    backbone.trainable = True
    for layer in backbone.layers[:-2]:
        layer.trainable = False

    text_input   = keras.Input(shape=(), dtype=tf.string, name="text")
    token_ids, padding_mask = preprocessor(text_input)
    seq_out      = backbone(token_ids, padding_mask=padding_mask)
    cls_token    = seq_out[:, 0, :]     # [CLS]

    x   = layers.Dropout(0.3)(cls_token)
    x   = layers.Dense(256, activation="gelu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(num_labels, activation="sigmoid", name="toxicity_scores")(x)

    model = keras.Model(inputs=text_input, outputs=out,
                        name="DistilBERTToxicityDetector")
    return model


# ── Build tf.data.Dataset ─────────────────────────────────────────────────────

def make_dataset(texts, labels, batch_size: int, shuffle: bool = False):
    """texts: list of raw strings; labels: float32 array [N, 6]"""
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000, seed=RANDOM_STATE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Custom weighted binary cross-entropy ─────────────────────────────────────

def make_weighted_bce(pos_weights: np.ndarray):
    """Returns a loss function that up-weights rare positive classes."""
    pos_w = tf.constant(pos_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(
            pos_w * y_true * tf.math.log(y_pred)
            + (1 - y_true) * tf.math.log(1 - y_pred)
        )
        return tf.reduce_mean(bce)

    return loss_fn


# ── Per-label AUC metric ──────────────────────────────────────────────────────

class MacroAUC(keras.metrics.Metric):
    def __init__(self, num_labels=6, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.aucs = [
            keras.metrics.AUC(name=f"auc_{i}") for i in range(num_labels)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i, auc in enumerate(self.aucs):
            auc.update_state(y_true[:, i], y_pred[:, i], sample_weight)

    def result(self):
        return tf.reduce_mean([auc.result() for auc in self.aucs])

    def reset_state(self):
        for auc in self.aucs:
            auc.reset_state()




# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_model(model, X_ids_test, X_mask_test, y_test, threshold: float = 0.5):
    print("\n[INFO] Evaluating toxicity detector on test set...")
    test_ds = make_dataset(X_ids_test, X_mask_test, y_test, batch_size=64)
    y_pred  = model.predict(test_ds, verbose=1)

    y_bin = (y_pred >= threshold).astype(int)

    print("\n" + "="*55)
    print("  Toxicity Detector — Per-label Report")
    print("="*55)
    print(classification_report(
        y_test, y_bin, target_names=LABEL_COLS, zero_division=0
    ))

    # Per-label AUC
    aucs, f1s = {}, {}
    for i, label in enumerate(LABEL_COLS):
        if y_test[:, i].sum() > 0:
            aucs[label] = round(roc_auc_score(y_test[:, i], y_pred[:, i]), 4)
            f1s[label]  = round(f1_score(y_test[:, i], y_bin[:, i], zero_division=0), 4)
        else:
            aucs[label] = None
            f1s[label]  = None

    macro_auc = np.mean([v for v in aucs.values() if v is not None])
    macro_f1  = np.mean([v for v in f1s.values()  if v is not None])

    print(f"\n  Macro AUC: {macro_auc:.4f}   Macro F1: {macro_f1:.4f}")
    print("\n  Per-label AUC:")
    for label, auc in aucs.items():
        print(f"    {label:<18}: {auc}")

    # Plot per-label AUC bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    valid_labels = [l for l in LABEL_COLS if aucs[l] is not None]
    valid_aucs   = [aucs[l] for l in valid_labels]
    bars = ax.barh(valid_labels, valid_aucs, color="#4c6ef5")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_xlim(0, 1)
    ax.set_title("Toxicity Detector — Per-label AUC")
    ax.set_xlabel("AUC")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(METRICS_DIR, "toxicity_per_label_auc.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"[INFO] AUC plot saved → {fig_path}")

    return {
        "macro_auc": round(macro_auc, 4),
        "macro_f1":  round(macro_f1,  4),
        "per_label_auc": aucs,
        "per_label_f1":  f1s,
    }


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],          label="Train Loss")
    axes[0].plot(history.history.get("val_loss",  []), label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history.get("macro_auc", []),     label="Train AUC")
    axes[1].plot(history.history.get("val_macro_auc", []), label="Val AUC")
    axes[1].set_title("Macro AUC")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "toxicity_training_history.png"), dpi=120)
    plt.close()
    print("[INFO] Training history plot saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of training data to use (0.0-1.0)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # ── Load data (text strings saved by preprocess_text.py) ─────────────────
    print("[INFO] Loading preprocessed text data...")
    import json as _json
    with open(os.path.join(DATA_DIR, "text_splits.json")) as f:
        splits = _json.load(f)
    texts_train = splits["train"]
    texts_val   = splits["val"]
    texts_test  = splits["test"]

    y_train      = np.load(os.path.join(DATA_DIR, "text_y_train.npy"))
    y_val        = np.load(os.path.join(DATA_DIR, "text_y_val.npy"))
    y_test       = np.load(os.path.join(DATA_DIR, "text_y_test.npy"))
    pos_weights  = np.load(os.path.join(DATA_DIR, "text_pos_weights.npy"))

    # Optional subsampling
    if args.sample < 1.0:
        n   = int(len(texts_train) * args.sample)
        idx = np.random.RandomState(RANDOM_STATE).choice(len(texts_train), n, replace=False)
        texts_train = [texts_train[i] for i in idx]
        y_train     = y_train[idx]
        print(f"[INFO] Using {args.sample*100:.0f}% of training data -> {n:,} samples")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Fine-tuning DistilBERT (keras_hub) for {args.epochs} epochs...")
    model, history = train_model(
        texts_train, y_train, texts_val, y_val,
        pos_weights=pos_weights,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    plot_history(history)

    # ── Save (.keras format required by Keras 3.x) ────────────────────────────
    save_path = os.path.join(MODELS_DIR, "toxicity_detector_tf.keras")
    model.save(save_path)
    print(f"[INFO] Model saved -> {save_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, texts_test, y_test)

    metrics_path = os.path.join(METRICS_DIR, "toxicity_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Toxicity metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
