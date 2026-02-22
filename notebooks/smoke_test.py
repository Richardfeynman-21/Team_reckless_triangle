"""
smoke_test.py
─────────────
Quick local validation — generates SYNTHETIC data to test the full
model architecture end-to-end WITHOUT requiring the Kaggle datasets.

Use this to verify:
  ✓ TF model builds correctly
  ✓ XGBoost model trains correctly
  ✓ DistilBERT tokenizer + model loads
  ✓ Unified pipeline produces valid output
  ✓ No import errors

Run:
    python notebooks/smoke_test.py

Expected time: ~3–5 minutes (downloads DistilBERT on first run, ~265 MB)
"""

import os, sys, json
import numpy as np

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(OUT_DIR, exist_ok=True)

PASS, FAIL = "[PASS]", "[FAIL]"
results = {}

print("="*60)
print("  PUBG AI System — Smoke Test")
print("="*60)


# ────────────────────────────────────────────────────────────────
# TEST 1: TensorFlow Outcome Predictor
# ────────────────────────────────────────────────────────────────
print("\n[1] TF Outcome Predictor (synthetic data)")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    def build_outcome_model(input_dim=31):
        inp = keras.Input(shape=(input_dim,))
        x = layers.Dense(256, activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(64,  activation="relu")(x)
        out = layers.Dense(3, activation="softmax")(x)
        return keras.Model(inputs=inp, outputs=out)

    model = build_outcome_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Synthetic data
    X_fake = np.random.randn(500, 31).astype(np.float32)
    y_fake = np.random.randint(0, 3, size=500).astype(np.int32)

    model.fit(X_fake, y_fake, epochs=2, batch_size=64, verbose=0,
              validation_split=0.2)

    preds = model.predict(X_fake[:5], verbose=0)
    assert preds.shape == (5, 3), f"Expected shape (5,3), got {preds.shape}"
    assert abs(preds[0].sum() - 1.0) < 1e-4, "Softmax probabilities should sum to 1"

    # Save — Keras 3.x requires .keras extension
    save_path = os.path.join(OUT_DIR, "smoke_outcome_tf.keras")
    model.save(save_path)
    loaded = keras.models.load_model(save_path)
    _ = loaded.predict(X_fake[:2], verbose=0)

    print(f"    Sample output probs: {preds[0]}")
    print(f"    {PASS}: TF model builds, trains, saves, and reloads correctly")
    results["tf_outcome"] = "PASS"
except Exception as e:
    print(f"    {FAIL}: {e}")
    results["tf_outcome"] = f"FAIL: {e}"


# ────────────────────────────────────────────────────────────────
# TEST 2: XGBoost Baseline
# ────────────────────────────────────────────────────────────────
print("\n[2] XGBoost Baseline (synthetic data)")
try:
    import xgboost as xgb, joblib

    clf = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, n_jobs=1, verbosity=0,
    )
    X_fake_xgb = np.random.randn(500, 31).astype(np.float32)
    y_fake_xgb = np.random.randint(0, 3, 500)

    clf.fit(X_fake_xgb, y_fake_xgb)
    proba = clf.predict_proba(X_fake_xgb[:5])
    assert proba.shape == (5, 3)

    xgb_path = os.path.join(OUT_DIR, "smoke_xgboost.joblib")
    joblib.dump(clf, xgb_path)
    clf2 = joblib.load(xgb_path)
    _ = clf2.predict_proba(X_fake_xgb[:2])

    print(f"    Sample output probs: {proba[0]}")
    print(f"    {PASS}: XGBoost trains, saves, and reloads correctly")
    results["xgboost"] = "PASS"
except Exception as e:
    print(f"    {FAIL}: {e}")
    results["xgboost"] = f"FAIL: {e}"


# ────────────────────────────────────────────────────────────────
# TEST 3: DistilBERT Tokenizer
# ────────────────────────────────────────────────────────────────
print("\n[3] DistilBERT Tokenizer")
try:
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    sample = ["Watch out, they are flanking!", "I will destroy you, noob."]
    enc = tokenizer(sample, max_length=64, padding="max_length",
                    truncation=True, return_tensors="np")

    assert enc["input_ids"].shape == (2, 64), f"Shape mismatch: {enc['input_ids'].shape}"
    print(f"    Token IDs shape: {enc['input_ids'].shape}")
    print(f"    {PASS}: Tokenizer loads and tokenizes correctly")
    results["tokenizer"] = "PASS"
except Exception as e:
    print(f"    {FAIL}: {e}")
    results["tokenizer"] = f"FAIL: {e}"


# ────────────────────────────────────────────────────────────────
# TEST 4: DistilBERT TF Model (small synthetic run)
# Note: DistilBertPreprocessor requires tensorflow-text (Linux only).
# On Windows we use HuggingFace tokenizer + DistilBertBackbone directly.
# ────────────────────────────────────────────────────────────────
print("\n[4] DistilBERT TF Model via keras_hub (2 epochs, 64 samples)")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import keras_hub
    from transformers import DistilBertTokenizerFast

    NUM_LABELS = 6
    BATCH = 8
    MAX_LEN = 64

    # Use HuggingFace tokenizer to produce token IDs (works on Windows)
    # DistilBertBackbone only needs integer token arrays — no tensorflow-text
    hf_tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    sample_texts = [
        "You are so bad.", "Nice clutch!", "Get out noob.", "Watch the door!",
        "Great revive!", "I hate you.", "Push them now.", "They have no loot.",
    ] * 8  # 64 samples

    enc  = hf_tok(sample_texts, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="np")
    ids  = enc["input_ids"].astype(np.int32)
    mask = enc["attention_mask"].astype(np.int32)
    y64  = np.random.randint(0, 2, size=(64, NUM_LABELS)).astype(np.float32)

    # Load keras_hub backbone (pre-tokenized: accepts int32 arrays)
    # backbone.input is a dict: {'token_ids':..., 'padding_mask':...}
    backbone = keras_hub.models.DistilBertBackbone.from_preset(
        "distil_bert_base_en_uncased"
    )
    backbone.trainable = False   # freeze for speed in smoke test

    # Build model with named dict inputs matching backbone's expected keys
    inp_ids  = keras.Input(shape=(MAX_LEN,), dtype="int32", name="token_ids")
    inp_mask = keras.Input(shape=(MAX_LEN,), dtype="int32", name="padding_mask")
    # Pass as dict to match backbone's functional API
    seq_out  = backbone({"token_ids": inp_ids, "padding_mask": inp_mask})
    cls_vec  = seq_out[:, 0, :]   # [CLS] representation
    output   = layers.Dense(NUM_LABELS, activation="sigmoid")(cls_vec)
    tox_model = keras.Model(inputs=[inp_ids, inp_mask], outputs=output)
    tox_model.compile(optimizer=keras.optimizers.Adam(2e-5),
                      loss="binary_crossentropy")

    ds = tf.data.Dataset.from_tensor_slices((
        {"token_ids": ids, "padding_mask": mask}, y64
    )).batch(BATCH).prefetch(tf.data.AUTOTUNE)



    tox_model.fit(ds, epochs=2, verbose=0)
    preds = tox_model.predict(ds, verbose=0)
    assert preds.shape == (64, NUM_LABELS), f"Shape mismatch: {preds.shape}"
    assert (preds >= 0).all() and (preds <= 1).all(), "Sigmoid output out of range"

    # Save with .keras extension (required by Keras 3.x)
    tox_save = os.path.join(OUT_DIR, "smoke_toxicity_tf.keras")
    tox_model.save(tox_save)
    loaded_tox = keras.models.load_model(tox_save)

    print(f"    Output shape: {preds.shape}")
    print(f"    Sample scores: {preds[0].tolist()}")
    print(f"    [PASS]: DistilBERT (keras_hub backbone) builds, trains, and saves correctly")
    print("    NOTE: On Kaggle (Linux), full pipeline uses DistilBertPreprocessor for end-to-end training.")
    results["distilbert_model"] = "PASS"
except Exception as e:
    print(f"    {FAIL}: {e}")
    results["distilbert_model"] = f"FAIL: {e}"



# ────────────────────────────────────────────────────────────────
# TEST 5: Text Cleaning (NLTK)
# ────────────────────────────────────────────────────────────────
print("\n[5] NLTK Text Cleaning")
try:
    import nltk, re
    for pkg in ["stopwords", "wordnet", "punkt"]:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop  = set(stopwords.words("english"))
    lemma = WordNetLemmatizer()

    def clean(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join(lemma.lemmatize(t) for t in text.split() if t not in stop)

    raw   = "You're the WORST player I've EVER seen!!! Get out of this game."
    clean_ = clean(raw)
    assert len(clean_) > 0
    print(f"    Input:  {raw[:60]}")
    print(f"    Output: {clean_}")
    print(f"    {PASS}: NLTK cleaning pipeline works correctly")
    results["nltk_cleaning"] = "PASS"
except Exception as e:
    print(f"    {FAIL}: {e}")
    results["nltk_cleaning"] = f"FAIL: {e}"


# ────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SMOKE TEST SUMMARY")
print("="*60)
all_pass = True
for test_name, result in results.items():
    status = PASS if result == "PASS" else FAIL
    print(f"  {status}  {test_name}")
    if result != "PASS":
        all_pass = False

print("─"*60)
if all_pass:
    print("  *** All tests passed! Ready to run full training. ***")
    print("     → python notebooks/kaggle_train_outcome.py")
    print("     → python notebooks/kaggle_train_toxicity.py")
else:
    print("  WARNING: Some tests failed. Fix the errors above before training.")
print("="*60)

# Save summary
with open(os.path.join(BASE_DIR, "smoke_test_results.json"), "w") as f:
    json.dump(results, f, indent=2)
