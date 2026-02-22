"""
outcome_predictor.py
────────────────────
Trains a TensorFlow Dense Neural Network (+ XGBoost baseline)
to predict PUBG match outcome: 0=Early Elim, 1=Top-10, 2=Victory.

Prerequisites:
    python data/preprocess_gameplay.py   (must be run first)

Usage:
    python models/outcome_predictor.py              # full training
    python models/outcome_predictor.py --sample 0.1 --epochs 5   # fast test
"""

import os
import sys
import argparse
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── TensorFlow ─────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, ConfusionMatrixDisplay,
)
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

LABEL_NAMES = ["Early Elim", "Top-10", "Victory"]
RANDOM_STATE = 42


# ── Build TF Model ─────────────────────────────────────────────────────────────

def build_tf_model(input_dim: int, num_classes: int = 3) -> keras.Model:
    """Deep tabular neural network with residual-style skip connection."""
    inp = keras.Input(shape=(input_dim,), name="gameplay_features")

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    out = layers.Dense(num_classes, activation="softmax", name="outcome_prob")(x)

    model = keras.Model(inputs=inp, outputs=out, name="PUBGOutcomePredictor")
    return model


# ── Train TF model ─────────────────────────────────────────────────────────────

def train_tf_model(X_train, y_train, X_val, y_val,
                   class_weights: dict, epochs: int = 60, batch_size: int = 2048):
    input_dim = X_train.shape[1]
    model = build_tf_model(input_dim)
    model.summary()

    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=len(X_train) // batch_size * 10,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "outcome_predictor_best.keras"),
            save_best_only=True, monitor="val_loss", verbose=0,
            save_format="keras",
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(BASE_DIR, "logs", "outcome_predictor"), histogram_freq=0
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )
    return model, history


# ── Train XGBoost baseline ────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val, class_weights: dict):
    print("\n[INFO] Training XGBoost baseline...")

    # Map sample weights
    w = np.array([class_weights[int(yi)] for yi in y_train])

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    clf.fit(
        X_train, y_train,
        sample_weight=w,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    joblib.dump(clf, os.path.join(MODELS_DIR, "outcome_xgboost.joblib"))
    print("[INFO] XGBoost model saved.")
    return clf


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_model(name: str, y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    f1  = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")

    print(f"\n{'='*50}")
    print(f"  {name} — Evaluation Results")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  Macro AUC: {auc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    fig_path = os.path.join(METRICS_DIR, f"confusion_{name.lower().replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {fig_path}")

    return {"f1_macro": round(f1, 4), "auc_macro": round(auc, 4)}


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "outcome_training_history.png"), dpi=120)
    plt.close()
    print("[INFO] Training history plot saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of training data to use (e.g. 0.1 for fast test)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[INFO] Loading processed gameplay data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    cw      = joblib.load(os.path.join(DATA_DIR, "class_weights.joblib"))

    # Optional subsampling for fast mode
    if args.sample < 1.0:
        n = int(len(X_train) * args.sample)
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), n, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"[INFO] Using {args.sample*100:.0f}% of training data → {n:,} rows")

    # ── Train TF model ────────────────────────────────────────────────────────
    print("\n[INFO] Training TensorFlow Neural Network...")
    tf_model, history = train_tf_model(
        X_train, y_train, X_val, y_val,
        class_weights=cw,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    plot_training_history(history)

    # Save TF model (.keras format, required by Keras 3.x)
    tf_save_path = os.path.join(MODELS_DIR, "outcome_predictor_tf.keras")
    tf_model.save(tf_save_path)
    print(f"[INFO] TF model saved → {tf_save_path}")

    # ── Evaluate TF model ─────────────────────────────────────────────────────
    y_pred_tf = tf_model.predict(X_test, batch_size=4096, verbose=0)
    tf_metrics = evaluate_model("TF Neural Network", y_test, y_pred_tf)

    # ── Train & evaluate XGBoost ──────────────────────────────────────────────
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, cw)
    y_pred_xgb = xgb_model.predict_proba(X_test)
    xgb_metrics = evaluate_model("XGBoost", y_test, y_pred_xgb)

    # ── Save combined metrics ─────────────────────────────────────────────────
    all_metrics = {
        "tf_neural_network": tf_metrics,
        "xgboost_baseline":  xgb_metrics,
    }
    metrics_path = os.path.join(METRICS_DIR, "outcome_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[DONE] All metrics saved → {metrics_path}")
    print(f"       TF  → F1: {tf_metrics['f1_macro']}, AUC: {tf_metrics['auc_macro']}")
    print(f"       XGB → F1: {xgb_metrics['f1_macro']}, AUC: {xgb_metrics['auc_macro']}")


if __name__ == "__main__":
    main()
