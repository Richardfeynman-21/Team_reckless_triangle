"""
preprocess_gameplay.py
──────────────────────
Preprocesses the PUBG Finish Placement dataset.

Expected input:  data/raw/train_V2.csv   (from Kaggle PUBG competition)
Outputs:
  - data/processed/X_train.npy
  - data/processed/X_val.npy
  - data/processed/X_test.npy
  - data/processed/y_train.npy  (3-class labels)
  - data/processed/y_val.npy
  - data/processed/y_test.npy
  - data/processed/scaler.joblib
  - data/processed/class_weights.joblib
  - data/processed/feature_names.txt

Run:
    python data/preprocess_gameplay.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_PATH     = os.path.join(os.path.dirname(__file__), "raw", "train_V2.csv")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
VAL_SIZE     = 0.10
TEST_SIZE    = 0.10

# Percentile cap for outlier clipping
OUTLIER_PCT  = 99

NUMERIC_COLS = [
    "assists", "boosts", "damageDealt", "DBNOs",
    "headshotKills", "heals", "killPlace", "killPoints",
    "kills", "killStreaks", "longestKill", "matchDuration",
    "maxPlace", "numGroups", "rankPoints", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills",
    "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Raw shape: {df.shape}")
    return df


def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with >50% missing values and known invalid entries."""
    before = len(df)
    df = df.dropna(thresh=int(df.shape[1] * 0.5))
    # Players with 0 matchDuration are corrupt entries
    df = df[df["matchDuration"] > 0]
    print(f"[INFO] Dropped {before - len(df)} invalid rows → {len(df)} remain")
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute remaining nulls in numeric columns."""
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def clip_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Cap values at the 99th percentile to dampen extreme outliers."""
    for col in cols:
        if col in df.columns:
            cap = df[col].quantile(OUTLIER_PCT / 100)
            df[col] = df[col].clip(upper=cap)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived gameplay features."""
    eps = 1e-6

    # Aggression: how offensive is this player?
    df["aggression_score"] = (
        df["kills"] + df["assists"] + df["damageDealt"] / 100
    )

    # Survival efficiency: healing & movement combined
    df["survival_efficiency"] = (
        df["heals"] + df["boosts"] + df["walkDistance"] / 100
    )

    # Combat ratio: kills relative to total distance (aggression vs. passive)
    total_dist = df["walkDistance"] + df["rideDistance"] + df["swimDistance"] + eps
    df["combat_ratio"] = df["kills"] / total_dist

    # Headshot skill
    df["headshot_rate"] = df["headshotKills"] / (df["kills"] + eps)

    # Team contribution
    df["team_contribution"] = (
        df["assists"] + df["revives"] + df["DBNOs"]
    )

    # Distance dominance: how much did they explore?
    df["total_distance"] = (
        df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    )

    # Normalized kill place (lower = better)
    df["normalized_kill_place"] = df["killPlace"] / (df["numGroups"] + eps)

    return df


def make_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket winPlacePerc into 3 outcome classes:
      0 = Early Elimination  (bottom 50%: winPlacePerc < 0.50)
      1 = Top-10 Finish      (50–90%:    0.50 ≤ winPlacePerc < 0.90)
      2 = Victory            (top 10%:   winPlacePerc ≥ 0.90)
    """
    df = df.dropna(subset=["winPlacePerc"])

    def _bucket(x):
        if x < 0.50:
            return 0
        elif x < 0.90:
            return 1
        else:
            return 2

    df["outcome"] = df["winPlacePerc"].apply(_bucket)
    dist = df["outcome"].value_counts().sort_index()
    print("[INFO] Class distribution:")
    labels = {0: "Early Elim", 1: "Top-10", 2: "Victory"}
    for cls, count in dist.items():
        print(f"       {labels[cls]} ({cls}): {count:,}  ({count/len(df)*100:.1f}%)")
    return df


def select_features(df: pd.DataFrame):
    """Return feature matrix X and label vector y."""
    engineered = [
        "aggression_score", "survival_efficiency", "combat_ratio",
        "headshot_rate", "team_contribution", "total_distance",
        "normalized_kill_place",
    ]
    feature_cols = [c for c in NUMERIC_COLS if c in df.columns] + engineered
    feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate

    X = df[feature_cols].values.astype(np.float32)
    y = df["outcome"].values.astype(np.int32)
    return X, y, feature_cols


def compute_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    print(f"[INFO] Class weights: {cw}")
    return cw


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_raw(RAW_PATH)
    df = drop_invalid(df)
    df = impute(df)
    df = clip_outliers(df, NUMERIC_COLS)
    df = engineer_features(df)
    df = make_outcome_label(df)

    X, y, feature_cols = select_features(df)
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # ── Train / Val / Test split ───────────────────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=VAL_SIZE + TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_tmp
    )
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Class weights ─────────────────────────────────────────────────────────
    cw = compute_weights(y_train)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(cw,     os.path.join(OUT_DIR, "class_weights.joblib"))

    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print("[DONE] Gameplay preprocessing complete. Files saved to data/processed/")


if __name__ == "__main__":
    main()
