"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        PUBG MATCH OUTCOME PREDICTOR — KAGGLE TRAINING SCRIPT               ║
║  Self-contained: Preprocessing → TF Neural Network → XGBoost → Evaluation  ║
║                                                                              ║
║  KAGGLE DATASET PATHS (add both to your notebook):                          ║
║    /kaggle/input/pubg-finish-placement-prediction/train_V2.csv              ║
║                                                                              ║
║  GPU: Not required (but speeds up TF training ~5x)                          ║
║  RAM: ~6 GB                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
#  SECTION 0 — INSTALL / IMPORTS
# ═══════════════════════════════════════════════════════════════════
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — prevents GUI crash on headless runs
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
print(f"TensorFlow version : {tf.__version__}")
print(f"GPUs available     : {tf.config.list_physical_devices('GPU')}")
print(f"XGBoost version    : {xgb.__version__}")

# ─── Output directory (works on Kaggle + locally) ─────────────────
OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath("."))), "saved_models"
)
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE  = 42
OUTCOME_NAMES = ["Early Elimination", "Top-10 Finish", "Victory"]


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════
# Kaggle path
KAGGLE_PATH = "/kaggle/input/pubg-finish-placement-prediction/train_V2.csv"
# Local path: data is in the pubg-finish-placement-prediction folder
LOCAL_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath("."))),
    "pubg-finish-placement-prediction", "train_V2.csv"
)
DATA_PATH   = KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else LOCAL_PATH

print(f"\n[1] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"    Raw shape: {df.shape}")
print(f"\n{df.dtypes.value_counts()}")
df.head(3)


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════════
print("\n[2] EDA")

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"    Columns with nulls:\n{missing}")

# Key stat distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
key_cols = ["kills", "damageDealt", "walkDistance", "heals",
            "boosts", "winPlacePerc", "rideDistance", "matchDuration"]
for i, col in enumerate(key_cols):
    if col in df.columns:
        axes[i].hist(df[col].dropna(), bins=50, color="#4c6ef5", alpha=0.8, edgecolor="none")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
plt.suptitle("PUBG Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_distributions.png"), dpi=120, bbox_inches="tight")
plt.show()
print("    EDA plot saved → eda_distributions.png")

# Correlation heatmap
numeric_df = df[key_cols].dropna()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_correlation.png"), dpi=120, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — PREPROCESSING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Preprocessing")

NUMERIC_COLS = [
    "assists", "boosts", "damageDealt", "DBNOs",
    "headshotKills", "heals", "killPlace", "killPoints",
    "kills", "killStreaks", "longestKill", "matchDuration",
    "maxPlace", "numGroups", "rankPoints", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills",
    "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints",
]

# ── 3a. Drop invalid rows ──────────────────────────────────────────
before = len(df)
df = df.dropna(thresh=int(df.shape[1] * 0.5))   # >50% missing → drop
df = df[df["matchDuration"] > 0]                  # corrupt rows
print(f"    Dropped {before - len(df):,} invalid rows → {len(df):,} remain")

# ── 3b. Median impute ─────────────────────────────────────────────
for col in NUMERIC_COLS:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# ── 3c. Outlier clipping (99th percentile) ────────────────────────
for col in NUMERIC_COLS:
    if col in df.columns:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)
print("    Outlier clipping applied at 99th percentile")

# ── 3d. Feature Engineering ───────────────────────────────────────
eps = 1e-6

df["aggression_score"] = df["kills"] + df["assists"] + df["damageDealt"] / 100

df["survival_efficiency"] = df["heals"] + df["boosts"] + df["walkDistance"] / 100

total_dist = df["walkDistance"] + df["rideDistance"] + df["swimDistance"] + eps
df["combat_ratio"] = df["kills"] / total_dist

df["headshot_rate"] = df["headshotKills"] / (df["kills"] + eps)

df["team_contribution"] = df["assists"] + df["revives"] + df["DBNOs"]

df["total_distance"] = total_dist - eps

df["normalized_kill_place"] = df["killPlace"] / (df["numGroups"] + eps)

print("    Engineered features: aggression_score, survival_efficiency, combat_ratio,")
print("                         headshot_rate, team_contribution, total_distance, normalized_kill_place")

# ── 3e. Create outcome label ──────────────────────────────────────
df = df.dropna(subset=["winPlacePerc"])

def make_label(x):
    if x < 0.50: return 0   # Early Elimination
    elif x < 0.90: return 1  # Top-10 Finish
    else: return 2            # Victory

df["outcome"] = df["winPlacePerc"].apply(make_label)

print("\n    Class distribution:")
dist = df["outcome"].value_counts().sort_index()
for cls, cnt in dist.items():
    print(f"      {OUTCOME_NAMES[cls]:20s}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

# Class distribution bar plot
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#da3633", "#388bfd", "#2ea043"]
bars = ax.bar(OUTCOME_NAMES, dist.values, color=colors, width=0.5)
ax.bar_label(bars, [f"{v:,}" for v in dist.values], padding=5)
ax.set_ylabel("Count")
ax.set_title("Outcome Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_class_dist.png"), dpi=120, bbox_inches="tight")
plt.show()

# ── 3f. Select features & split ───────────────────────────────────
ENGINEERED = [
    "aggression_score", "survival_efficiency", "combat_ratio",
    "headshot_rate", "team_contribution", "total_distance", "normalized_kill_place",
]
FEATURE_COLS = [c for c in NUMERIC_COLS if c in df.columns] + ENGINEERED
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))   # deduplicate

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["outcome"].values.astype(np.int32)

print(f"\n    Feature matrix: {X.shape}  →  {len(FEATURE_COLS)} features")

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp
)
print(f"    Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── 3g. Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR, "gameplay_scaler.joblib"))
print("    StandardScaler fitted and saved → gameplay_scaler.joblib")

# ── 3h. Class weights ─────────────────────────────────────────────
classes   = np.unique(y_train)
weights   = compute_class_weight("balanced", classes=classes, y=y_train)
cw        = dict(zip(classes.tolist(), weights.tolist()))
print(f"    Class weights: {cw}")
joblib.dump(cw, os.path.join(OUT_DIR, "gameplay_class_weights.joblib"))

# Save feature names for pipeline
with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
    f.write("\n".join(FEATURE_COLS))


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — TENSORFLOW NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Training TensorFlow Neural Network")

def build_model(input_dim):
    inp = keras.Input(shape=(input_dim,), name="gameplay_input")

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

    out = layers.Dense(3, activation="softmax", name="outcome_prob")(x)
    return keras.Model(inputs=inp, outputs=out, name="PUBGOutcomePredictor")

BATCH_SIZE = 2048
EPOCHS     = 60

model = build_model(X_train.shape[1])
model.summary()

lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=(len(X_train) // BATCH_SIZE) * 10,
)

model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb_list = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(OUT_DIR, "outcome_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=cw,
    callbacks=cb_list,
    verbose=1,
)

# ── Training curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history.history["loss"],     label="Train", color="#4c6ef5")
axes[0].plot(history.history["val_loss"], label="Val",   color="#ff6b6b")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
axes[1].plot(history.history["accuracy"],     label="Train", color="#4c6ef5")
axes[1].plot(history.history["val_accuracy"], label="Val",   color="#ff6b6b")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
plt.suptitle("TF Outcome Predictor — Training History", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "outcome_tf_history.png"), dpi=120, bbox_inches="tight")
plt.show()

# ── Save TF model ─────────────────────────────────────────────────
tf_save_path = os.path.join(OUT_DIR, "outcome_predictor_tf.keras")
model.save(tf_save_path)
print(f"    TF model saved → {tf_save_path}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 — XGBOOST BASELINE
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Training XGBoost Baseline")

sample_weights = np.array([cw[int(yi)] for yi in y_train])

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=20,
    verbosity=1,
)
xgb_clf.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=50,
)
joblib.dump(xgb_clf, os.path.join(OUT_DIR, "outcome_xgboost.joblib"))
print("    XGBoost saved → outcome_xgboost.joblib")

# Feature importance plot
fig, ax = plt.subplots(figsize=(9, 7))
importance = pd.Series(xgb_clf.feature_importances_, index=FEATURE_COLS)
importance.nlargest(20).sort_values().plot(
    kind="barh", ax=ax, color="#4c6ef5", edgecolor="none"
)
ax.set_title("XGBoost Feature Importance (Top 20)")
ax.set_xlabel("F-score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "xgb_feature_importance.png"), dpi=120, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 — EVALUATION
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Evaluation on Test Set")

def evaluate(name, y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    f1  = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(classification_report(y_true, y_pred, target_names=OUTCOME_NAMES))
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Macro AUC : {auc:.4f}")

    # Confusion matrix
    cm   = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=OUTCOME_NAMES, yticklabels=OUTCOME_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("+", "")
    plt.savefig(os.path.join(OUT_DIR, f"cm_{fname}.png"), dpi=120, bbox_inches="tight")
    plt.show()

    return {"f1_macro": round(f1, 4), "auc_macro": round(auc, 4)}

# TF evaluation
y_pred_tf  = model.predict(X_test, batch_size=4096, verbose=0)
tf_metrics = evaluate("TF Neural Network", y_test, y_pred_tf)

# XGBoost evaluation
y_pred_xgb  = xgb_clf.predict_proba(X_test)
xgb_metrics = evaluate("XGBoost Baseline", y_test, y_pred_xgb)

# Side-by-side comparison
fig, ax = plt.subplots(figsize=(8, 4))
models  = ["TF Neural Network", "XGBoost"]
f1s     = [tf_metrics["f1_macro"], xgb_metrics["f1_macro"]]
aucs    = [tf_metrics["auc_macro"], xgb_metrics["auc_macro"]]
x       = np.arange(len(models))
w       = 0.35
bars1 = ax.bar(x - w/2, f1s,  w, label="Macro F1",  color="#4c6ef5")
bars2 = ax.bar(x + w/2, aucs, w, label="Macro AUC", color="#2ea043")
ax.bar_label(bars1, [f"{v:.4f}" for v in f1s],  padding=3, fontsize=10)
ax.bar_label(bars2, [f"{v:.4f}" for v in aucs], padding=3, fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Outcome Predictor")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=120, bbox_inches="tight")
plt.show()

# ── Save all metrics ───────────────────────────────────────────────
metrics = {
    "tf_neural_network": tf_metrics,
    "xgboost_baseline":  xgb_metrics,
}
with open(os.path.join(OUT_DIR, "outcome_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'═'*55}")
print("  OUTCOME PREDICTOR TRAINING COMPLETE")
print(f"  TF  → F1: {tf_metrics['f1_macro']}  AUC: {tf_metrics['auc_macro']}")
print(f"  XGB → F1: {xgb_metrics['f1_macro']}  AUC: {xgb_metrics['auc_macro']}")
print(f"{'═'*55}")
print(f"\n  Files saved to: {OUT_DIR}/")
print("  - outcome_predictor_tf/   (TF SavedModel)")
print("  - outcome_xgboost.joblib  (XGBoost)")
print("  - gameplay_scaler.joblib  (Scaler)")
print("  - outcome_metrics.json    (Metrics)")
print("  - *.png                   (Plots)")
