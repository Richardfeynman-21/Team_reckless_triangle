"""
unified_pipeline.py
────────────────────
Single cohesive system that integrates:
  1. Gameplay Outcome Prediction  (TF Neural Network or XGBoost)
  2. Chat Toxicity Detection      (DistilBERT)
  3. Contextual Feedback Engine   (rule-based over model outputs)

Usage (standalone test):
    python pipeline/unified_pipeline.py --test

Returns a PlayerReport dataclass per player-session.
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import tensorflow as tf
from tensorflow import keras
import keras_hub
from transformers import DistilBertTokenizerFast

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

OUTCOME_LABELS     = ["Early Elimination", "Top-10 Finish", "Victory"]
TOXICITY_LABELS    = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TOXICITY_THRESHOLD = 0.5
DISTILBERT_ID      = "distilbert-base-uncased"
MAX_LEN            = 128

# Singleton tokenizer cache
_tokenizer = None
def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_ID)
    return _tokenizer


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ToxicityResult:
    message: str
    scores: Dict[str, float]         # label → sigmoid score
    is_toxic: bool
    severity: str                     # "clean" | "mild" | "severe"
    flagged_labels: List[str]


@dataclass
class PlayerReport:
    player_id: str
    outcome_probs: Dict[str, float]   # class_name → probability
    predicted_outcome: str
    toxicity_results: List[ToxicityResult]
    aggregate_toxicity: float         # avg max score across messages
    feedback: List[str]               # human-readable tactical tips


# ── Model loader (singleton cache) ────────────────────────────────────────────

class ModelRegistry:
    _outcome_tf  = None
    _outcome_xgb = None
    _toxicity_tf = None
    _scaler      = None   # tokenizer no longer needed; keras_hub handles it

    @classmethod
    def get_outcome_tf(cls):
        if cls._outcome_tf is None:
            path = os.path.join(MODELS_DIR, "outcome_predictor_tf.keras")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TF outcome model not found at {path}\n"
                    "Run: python models/outcome_predictor.py"
                )
            cls._outcome_tf = keras.models.load_model(path)
        return cls._outcome_tf

    @classmethod
    def get_outcome_xgb(cls):
        if cls._outcome_xgb is None:
            path = os.path.join(MODELS_DIR, "outcome_xgboost.joblib")
            if os.path.exists(path):
                cls._outcome_xgb = joblib.load(path)
        return cls._outcome_xgb

    @classmethod
    def get_toxicity(cls):
        if cls._toxicity_tf is None:
            path = os.path.join(MODELS_DIR, "toxicity_detector_tf.keras")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Toxicity model not found at {path}\n"
                    "Run: python models/toxicity_detector.py"
                )
            # compile=False: skips custom loss/metric re-registration;
            # model is inference-only so this is correct.
            cls._toxicity_tf = keras.models.load_model(path, compile=False)
        return cls._toxicity_tf

    @classmethod
    def get_scaler(cls):
        if cls._scaler is None:
            path = os.path.join(DATA_DIR, "scaler.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Scaler not found at {path}\n"
                    "Run: python data/preprocess_gameplay.py"
                )
            cls._scaler = joblib.load(path)
        return cls._scaler


# ── Gameplay prediction ───────────────────────────────────────────────────────

FEATURE_ORDER = [
    "assists", "boosts", "damageDealt", "DBNOs",
    "headshotKills", "heals", "killPlace", "killPoints",
    "kills", "killStreaks", "longestKill", "matchDuration",
    "maxPlace", "numGroups", "rankPoints", "revives",
    "rideDistance", "roadKills", "swimDistance", "teamKills",
    "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints",
    # engineered
    "aggression_score", "survival_efficiency", "combat_ratio",
    "headshot_rate", "team_contribution", "total_distance", "normalized_kill_place",
]


def _engineer(stats: dict) -> dict:
    """Add engineered features to a raw stats dict."""
    eps = 1e-6
    kills = stats.get("kills", 0)
    stats["aggression_score"] = (
        kills + stats.get("assists", 0) + stats.get("damageDealt", 0) / 100
    )
    stats["survival_efficiency"] = (
        stats.get("heals", 0) + stats.get("boosts", 0)
        + stats.get("walkDistance", 0) / 100
    )
    total_dist = (
        stats.get("walkDistance", 0)
        + stats.get("rideDistance", 0)
        + stats.get("swimDistance", 0)
        + eps
    )
    stats["combat_ratio"] = kills / total_dist
    stats["headshot_rate"] = stats.get("headshotKills", 0) / (kills + eps)
    stats["team_contribution"] = (
        stats.get("assists", 0)
        + stats.get("revives", 0)
        + stats.get("DBNOs", 0)
    )
    stats["total_distance"] = total_dist - eps
    num_groups = stats.get("numGroups", 1) or 1
    stats["normalized_kill_place"] = stats.get("killPlace", 0) / num_groups
    return stats


def predict_outcome(gameplay_stats: dict, use_xgb: bool = False) -> Dict[str, float]:
    """
    Args:
        gameplay_stats: dict of raw numeric gameplay stats
        use_xgb: if True, use XGBoost; else TF neural network

    Returns:
        dict {outcome_label: probability}
    """
    stats = _engineer(dict(gameplay_stats))
    x = np.array([[stats.get(f, 0.0) for f in FEATURE_ORDER]], dtype=np.float32)

    scaler = ModelRegistry.get_scaler()
    x = scaler.transform(x)

    if use_xgb:
        model = ModelRegistry.get_outcome_xgb()
        if model is None:
            raise RuntimeError("XGBoost model not trained yet.")
        probs = model.predict_proba(x)[0]
    else:
        model = ModelRegistry.get_outcome_tf()
        probs = model.predict(x, verbose=0)[0]   # keras model: use .predict()

    return {label: float(round(p, 4)) for label, p in zip(OUTCOME_LABELS, probs)}


# ── Toxicity prediction ───────────────────────────────────────────────────────

def _classify_severity(scores: Dict[str, float]) -> str:
    max_score = max(scores.values())
    if scores.get("severe_toxic", 0) > 0.5 or scores.get("threat", 0) > 0.5:
        return "severe"
    elif max_score >= TOXICITY_THRESHOLD:
        return "mild"
    return "clean"


def predict_toxicity(messages: List[str], threshold: float = TOXICITY_THRESHOLD) -> List[ToxicityResult]:
    """Tokenise messages and run the DistilBERT toxicity model.

    Args:
        messages:  List of raw chat strings.
        threshold: Score above which a label is flagged (default 0.5).
    """
    if not messages:
        return []

    # ── Tokenize ───────────────────────────────────────────────────────────────
    tokenizer = _get_tokenizer()
    enc = tokenizer(
        messages,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids  = enc["input_ids"].astype(np.int32)
    mask = enc["attention_mask"].astype(np.int32)

    # ── Predict ────────────────────────────────────────────────────────────────
    model = ModelRegistry.get_toxicity()
    preds = model.predict(
        {"token_ids": ids, "padding_mask": mask}, verbose=0
    )   # shape: [N, 6]

    # ── Build results ─────────────────────────────────────────────────────────
    results = []
    for i, msg in enumerate(messages):
        scores  = {
            label: float(round(float(preds[i, j]), 4))
            for j, label in enumerate(TOXICITY_LABELS)
        }
        flagged = [l for l, s in scores.items() if s >= threshold]
        results.append(ToxicityResult(
            message=msg,
            scores=scores,
            is_toxic=bool(flagged),
            severity=_classify_severity(scores),
            flagged_labels=flagged,
        ))
    return results


# ── Feedback Engine ───────────────────────────────────────────────────────────

def generate_feedback(
    outcome_probs: Dict[str, float],
    toxicity_results: List[ToxicityResult],
    gameplay_stats: dict,
) -> List[str]:
    """Rule-based feedback generator combining both model outputs."""
    feedback = []

    # ── Outcome-based tips ────────────────────────────────────────────────────
    elim_p  = outcome_probs.get("Early Elimination", 0)
    top10_p = outcome_probs.get("Top-10 Finish", 0)
    win_p   = outcome_probs.get("Victory", 0)

    if elim_p > 0.60:
        feedback.append(
            "⚠️ HIGH RISK: Early elimination probability is high. "
            "Stay in cover, avoid open fields, and play defensively."
        )
    if elim_p > 0.40 and gameplay_stats.get("walkDistance", 0) < 500:
        feedback.append(
            "🏃 Movement tip: You're barely moving. Staying stationary "
            "makes you an easy target — rotate and find better positioning."
        )
    if top10_p > 0.50:
        feedback.append(
            "💡 Good standing! Focus on positioning for the final circle. "
            "Conserve ammo and heal up before the mid-game push."
        )
    if win_p > 0.35:
        feedback.append(
            "🏆 Win scenario is realistic. Maintain aggressive pressure, "
            "coordinate with your squad, and secure the final zone."
        )

    # ── Combat-based tips ─────────────────────────────────────────────────────
    kills  = gameplay_stats.get("kills", 0)
    damage = gameplay_stats.get("damageDealt", 0)
    if kills == 0 and damage < 100:
        feedback.append(
            "🎯 Engagement tip: You have 0 kills and low damage. "
            "Try engaging enemies at range to contribute to the squad."
        )
    if gameplay_stats.get("heals", 0) == 0 and gameplay_stats.get("boosts", 0) == 0:
        feedback.append(
            "💊 Resource tip: No heals or boosts used. "
            "Loot medical supplies and use boosts proactively — they also give "
            "temporary speed during the final circles."
        )
    revives = gameplay_stats.get("revives", 0)
    if revives == 0 and gameplay_stats.get("DBNOs", 0) > 0:
        feedback.append(
            "🤝 Team tip: Teammates were downed but not revived. "
            "Prioritize revives when safe — squad size is your biggest advantage."
        )

    # ── Toxicity-based warnings ───────────────────────────────────────────────
    severe_msgs = [r for r in toxicity_results if r.severity == "severe"]
    mild_msgs   = [r for r in toxicity_results if r.severity == "mild"]

    if severe_msgs:
        feedback.append(
            f"🚨 SEVERE TOXICITY detected in {len(severe_msgs)} message(s). "
            "Threats or harassment flagged — this account may be reviewed for sanctions."
        )
    elif mild_msgs:
        feedback.append(
            f"🔴 Toxic language detected in {len(mild_msgs)} message(s). "
            "Please communicate respectfully — toxic behavior impacts team performance."
        )
    elif toxicity_results:
        feedback.append(
            "✅ All chat messages are clean. Great sportsmanship!"
        )

    if not feedback:
        feedback.append("📊 Performance looks steady. Keep up the solid play!")

    return feedback


# ── Main pipeline entry point ─────────────────────────────────────────────────

def analyze_player(
    player_id: str,
    gameplay_stats: dict,
    chat_messages: List[str],
    use_xgb: bool = False,
) -> PlayerReport:
    """
    Full pipeline: takes raw gameplay stats + chat messages,
    returns a structured PlayerReport.
    """
    outcome_probs     = predict_outcome(gameplay_stats, use_xgb=use_xgb)
    predicted_outcome = max(outcome_probs, key=outcome_probs.get)
    toxicity_results  = predict_toxicity(chat_messages)

    agg_toxicity = 0.0
    if toxicity_results:
        agg_toxicity = float(np.mean([
            max(r.scores.values()) for r in toxicity_results
        ]))

    feedback = generate_feedback(outcome_probs, toxicity_results, gameplay_stats)

    return PlayerReport(
        player_id=player_id,
        outcome_probs=outcome_probs,
        predicted_outcome=predicted_outcome,
        toxicity_results=toxicity_results,
        aggregate_toxicity=round(agg_toxicity, 4),
        feedback=feedback,
    )


# ── Test harness ──────────────────────────────────────────────────────────────

SAMPLE_STATS = {
    "assists": 2, "boosts": 3, "damageDealt": 420.5, "DBNOs": 1,
    "headshotKills": 1, "heals": 4, "killPlace": 8, "killPoints": 1200,
    "kills": 3, "killStreaks": 2, "longestKill": 87.3, "matchDuration": 1800,
    "maxPlace": 99, "numGroups": 95, "rankPoints": 1500, "revives": 1,
    "rideDistance": 500.0, "roadKills": 0, "swimDistance": 0.0, "teamKills": 0,
    "vehicleDestroys": 0, "walkDistance": 2100.0, "weaponsAcquired": 5, "winPoints": 1400,
}

SAMPLE_CHAT = [
    "Watch out, they're coming from the east!",
    "I will find you and destroy you, loser.",
    "Great teamwork everyone, let's go!",
    "You are so stupid, why did you go there?",
    "Cover me, I'm reviving Jake.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with sample data")
    args = parser.parse_args()

    if args.test:
        print("[TEST] Running unified pipeline with sample data...\n")
        report = analyze_player(
            player_id="player_001",
            gameplay_stats=SAMPLE_STATS,
            chat_messages=SAMPLE_CHAT,
        )
        print(json.dumps(asdict(report), indent=2))
    else:
        print("Use --test to run a pipeline test with sample data.")
        print("For production use, import 'analyze_player' from this module.")


if __name__ == "__main__":
    main()
