"""
colab_train_outcome.py
══════════════════════
Google Colab version of the PUBG Match Outcome Predictor training script.

INSTRUCTIONS FOR COLAB:
1. Upload this script to a Colab cell.
2. Go to: Runtime → Change runtime type → T4 GPU.
3. Upload 'train_V2.csv' (from PUBG Kaggle) directly to the '/content/' folder.
4. Run the cell. Your models and plots will be saved to your Google Drive.
"""

# ═══════════════════════════════════════════════════════════════════
#  SECTION 0 — GPU CHECK (must run first)
# ═══════════════════════════════════════════════════════════════════
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU detected: {tf.test.gpu_device_name()}")
else:
    print("⚠ NO GPU DETECTED. Training will be slow.")

# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — INSTALLS / IMPORTS / DRIVE MOUNT
# ═══════════════════════════════════════════════════════════════════
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = "/content/drive/MyDrive/pubg_ai/saved_models"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    OUT_DIR = DRIVE_DIR
    print(f"✓ Google Drive mounted. Savings to: {OUT_DIR}")
except ImportError:
    OUT_DIR = "saved_models"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"! Not on Colab. Saving to local: {OUT_DIR}")

warnings.filterwarnings("ignore")
RANDOM_STATE  = 42
OUTCOME_NAMES = ["Early Elimination", "Top-10 Finish", "Victory"]
DATA_PATH     = "/content/train_V2.csv" if os.path.exists("/content/train_V2.csv") else "train_V2.csv"

# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found! Please upload 'train_V2.csv' to {DATA_PATH}")

print(f"\n[1] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"    Raw shape: {df.shape}")

NUMERIC_COLS = [
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals",
    "killPlace", "killPoints", "kills", "killStreaks", "longestKill",
    "matchDuration", "maxPlace", "numGroups", "rankPoints", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills",
    "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints",
]

# Preprocess
df = df.dropna(thresh=int(df.shape[1] * 0.5))
df = df[df["matchDuration"] > 0]
for col in NUMERIC_COLS:
    df[col] = df[col].fillna(df[col].median())
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

# Feature Engineering
eps = 1e-6
df["aggression_score"] = df["kills"] + df["assists"] + df["damageDealt"] / 100
df["survival_efficiency"] = df["heals"] + df["boosts"] + df["walkDistance"] / 100
total_dist = df["walkDistance"] + df["rideDistance"] + df["swimDistance"] + eps
df["combat_ratio"] = df["kills"] / total_dist
df["headshot_rate"] = df["headshotKills"] / (df["kills"] + eps)
df["team_contribution"] = df["assists"] + df["revives"] + df["DBNOs"]
df["total_distance"] = total_dist - eps
df["normalized_kill_place"] = df["killPlace"] / (df["numGroups"] + eps)

# Labels
df = df.dropna(subset=["winPlacePerc"])
df["outcome"] = df["winPlacePerc"].apply(lambda x: 0 if x < 0.50 else 1 if x < 0.90 else 2)

# Select features & Split
ENGINEERED = ["aggression_score", "survival_efficiency", "combat_ratio", "headshot_rate", "team_contribution", "total_distance", "normalized_kill_place"]
FEATURE_COLS = list(dict.fromkeys([c for c in NUMERIC_COLS if c in df.columns] + ENGINEERED))
X = df[FEATURE_COLS].values.astype(np.float32)
y = df["outcome"].values.astype(np.int32)

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR, "gameplay_scaler.joblib"))

# Weights
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
cw = dict(zip(classes.tolist(), weights.tolist()))

# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL & TRAINING
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Training Neural Network")
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],), name="gameplay_input"),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax", name="outcome_prob")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "outcome_best.keras"), monitor="val_loss", save_best_only=True)
]

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=4096, class_weight=cw, callbacks=cb_list)
model.save(os.path.join(OUT_DIR, "outcome_predictor_tf.keras"))

# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — XGBOOST & EVALUATION
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Training XGBoost & Evaluating")
xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
joblib.dump(xgb_clf, os.path.join(OUT_DIR, "outcome_xgboost.joblib"))

y_pred_tf = model.predict(X_test, verbose=0)
f1_tf = f1_score(y_test, np.argmax(y_pred_tf, axis=1), average="macro")
auc_tf = roc_auc_score(y_test, y_pred_tf, multi_class="ovr", average="macro")

metrics = {"tf_neural_network": {"f1_macro": round(f1_tf, 4), "auc_macro": round(auc_tf, 4)}}
with open(os.path.join(OUT_DIR, "outcome_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ DONE! TF Model saved to: {OUT_DIR}/outcome_predictor_tf.keras")
print(f"  Macro AUC: {auc_tf:.4f}  |  Macro F1: {f1_tf:.4f}")
