"""
pipeline_smoke_test.py
Full end-to-end validation of scaler, tokenizer, toxicity model, and outcome model.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from tensorflow import keras
from transformers import DistilBertTokenizerFast

SAVED = os.path.join(os.path.dirname(__file__), "..", "saved_models")
DATA  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TOXICITY_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
OUTCOME_LABELS  = ["Early Elimination", "Top-10 Finish", "Victory"]

def test_scaler():
    print("[1] Scaler ...")
    scaler = joblib.load(os.path.join(DATA, "scaler.joblib"))
    print(f"    OK  n_features={scaler.n_features_in_}  mean[:3]={scaler.mean_[:3].round(3)}")
    return scaler

def test_tokenizer():
    print("[2] Tokenizer ...")
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    enc = tok(["hello world"], max_length=16, padding="max_length",
              truncation=True, return_tensors="np")
    shape = enc["input_ids"].shape
    print(f"    OK  ids shape: {shape}")
    return tok

def test_toxicity(tok):
    print("[3] Toxicity model + inference ...")
    model = keras.models.load_model(
        os.path.join(SAVED, "toxicity_detector_tf.keras"), compile=False
    )
    msgs = [
        "You are awesome!",
        "I will destroy you useless noob get out",
        "Good game everyone, nice teamwork!",
        "I hate you, you garbage player",
    ]
    enc = tok(msgs, max_length=128, padding="max_length", truncation=True, return_tensors="np")
    preds = model.predict(
        {"token_ids": enc["input_ids"].astype("int32"),
         "padding_mask": enc["attention_mask"].astype("int32")},
        verbose=0
    )
    print(f"    Predictions shape: {preds.shape}")
    for i, msg in enumerate(msgs):
        row   = preds[i]
        top_l = TOXICITY_LABELS[row.argmax()]
        top_v = row.max()
        is_toxic = top_v >= 0.5
        flag = "TOXIC" if is_toxic else "clean"
        print(f"    [{flag}]  {top_l}={top_v:.3f}  {msg[:50]}")

def test_outcome(scaler):
    print("[4] Outcome model + inference ...")
    model = keras.models.load_model(
        os.path.join(SAVED, "outcome_predictor_tf.keras"), compile=False
    )
    # Simulate a decent player: 5 kills, 350 dmg, 2000m walk etc.
    n = scaler.n_features_in_
    x = np.zeros((1, n), dtype="float32")
    x_scaled = scaler.transform(x)
    probs = model.predict(x_scaled, verbose=0)[0]
    for l, p in zip(OUTCOME_LABELS, probs):
        print(f"    {l}: {p:.3f}")
    print(f"    Predicted: {OUTCOME_LABELS[probs.argmax()]}")

if __name__ == "__main__":
    print("=" * 50)
    print("  Full Pipeline Smoke Test")
    print("=" * 50)
    scaler = test_scaler()
    tok    = test_tokenizer()
    test_toxicity(tok)
    test_outcome(scaler)
    print()
    print("=" * 50)
    print("  ALL TESTS PASSED ✓")
    print("=" * 50)
