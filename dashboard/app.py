"""
app.py
──────
PUBG AI Analytics Dashboard
Built with Streamlit.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Add project root to path ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ─── Cached model loaders — load ONCE at startup, shared across all sessions ──
@st.cache_resource(show_spinner="⏳ Loading Outcome model...")
def _load_outcome_model():
    from tensorflow import keras
    path = os.path.join(BASE_DIR, "saved_models", "outcome_predictor_tf.keras")
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner="⏳ Loading Toxicity model (473 MB — one-time)...")
def _load_toxicity_model():
    from tensorflow import keras
    path = os.path.join(BASE_DIR, "saved_models", "toxicity_detector_tf.keras")
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner="⏳ Loading scaler...")
def _load_scaler():
    import joblib
    path = os.path.join(BASE_DIR, "data", "processed", "scaler.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def _get_pipeline():
    from pipeline.unified_pipeline import analyze_player
    return analyze_player

def _get_toxicity_fn():
    from pipeline.unified_pipeline import predict_toxicity
    return predict_toxicity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PUBG AI Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Clean Light Mode Background */
.stApp {
    background: #f8fafc;
    color: #0f172a;
}

/* Base text color override for Streamlit */
.stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp div {
    color: #0f172a !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
}

/* Outcome badge */
.badge-win    { background: #dcfce7; color: #166534; border-radius: 8px; padding: 4px 12px; font-weight: 600; border: 1px solid #bbf7d0; }
.badge-top10  { background: #dbeafe; color: #1e40af; border-radius: 8px; padding: 4px 12px; font-weight: 600; border: 1px solid #bfdbfe; }
.badge-elim   { background: #fee2e2; color: #991b1b; border-radius: 8px; padding: 4px 12px; font-weight: 600; border: 1px solid #fecaca; }

/* Toxicity heatmap cells */
.tox-cell     { padding: 8px; border-radius: 6px; text-align: center; font-weight: 600; color: #0f172a !important; }

/* Override Streamlit elements to fit light theme */
.stButton>button {
    background: #2563eb !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
}

.stTextInput>div>div>input, .stNumberInput>div>div>input {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 6px !important;
}

.stTextArea>div>div>textarea {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
}

.stDataFrame, .stTable {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Force dataframe header text to be visible */
th {
    background: #f1f5f9 !important;
    color: #334155 !important;
}
td {
    color: #0f172a !important;
}

h1, h2, h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* Sidebar specifically */
section[data-testid="stSidebar"] p {
    color: #475569 !important;
    font-weight: 500;
}
section[data-testid="stSidebar"] [aria-selected="true"] p {
    color: #2563eb !important;
    font-weight: 700;
}

/* Markdown info boxes */
.stAlert {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 PUBG AI System")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "⚔️ Player Analysis", "💬 Chat Moderation", "📊 Model Metrics", "⚖️ Ethics"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Built with TensorFlow + DistilBERT<br>"
        "Datasets: PUBG Kaggle + Jigsaw</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🎯 PUBG Intelligence Dashboard")
    st.markdown(
        "<p style='color:#8b949e;font-size:1.05rem'>AI-powered system combining "
        "gameplay outcome prediction and real-time toxicity detection</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>⚔️ Outcome Predictor</h3>
            <p style='color:#8b949e;font-size:0.9rem'>
            Deep Neural Network + XGBoost trained on 4.4M PUBG matches.
            Predicts: Early Elimination, Top-10 Finish, or Victory.
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>💬 Toxicity Detector</h3>
            <p style='color:#8b949e;font-size:0.9rem'>
            DistilBERT fine-tuned on 160K Jigsaw comments.
            Detects: toxic, obscene, threat, insult, identity hate.
            </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>🔗 Unified Pipeline</h3>
            <p style='color:#8b949e;font-size:0.9rem'>
            Combines both models to generate per-player tactical
            feedback and safety warnings in real time.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    ```
    Gameplay Stats ──► TF Neural Network ──────────────────────┐
                    ──► XGBoost Baseline ──────────────────────►│ Unified Pipeline
    Chat Messages  ──► DistilBERT (NLP) ──────────────────────►│      │
                                                                └──────▼
                                                          Player Report + Feedback
    ```
    """)
    st.info("👈 Use the sidebar to navigate to **Player Analysis** or **Chat Moderation** to try the live models.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Player Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Player Analysis":
    st.markdown("# ⚔️ Player Analysis")
    st.markdown(
        "<p style='color:#8b949e'>Enter gameplay statistics to get outcome predictions and tactical feedback.</p>",
        unsafe_allow_html=True,
    )

    with st.form("player_form"):
        st.markdown("### Player Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kills       = st.number_input("Kills",        0, 50, 3)
            assists     = st.number_input("Assists",       0, 30, 2)
            damage      = st.number_input("Damage Dealt", 0.0, 5000.0, 420.5)
            headshotk   = st.number_input("Headshot Kills", 0, 20, 1)
        with col2:
            heals       = st.number_input("Heals Used",   0, 30, 4)
            boosts      = st.number_input("Boosts Used",  0, 20, 3)
            revives     = st.number_input("Revives",      0, 10, 1)
            dbnos       = st.number_input("DBNOs",        0, 20, 1)
        with col3:
            walk_dist   = st.number_input("Walk Distance (m)",  0.0, 15000.0, 2100.0)
            ride_dist   = st.number_input("Ride Distance (m)",  0.0, 50000.0, 500.0)
            swim_dist   = st.number_input("Swim Distance (m)",  0.0, 5000.0,  0.0)
            weapons     = st.number_input("Weapons Acquired",   0, 20, 5)
        with col4:
            kill_place  = st.number_input("Kill Place",    1, 100, 8)
            num_groups  = st.number_input("Num Groups",    1, 100, 95)
            match_dur   = st.number_input("Match Duration (s)", 100, 2500, 1800)
            kill_stk    = st.number_input("Kill Streaks",  0, 10, 2)

        st.markdown("### Chat Messages (optional)")
        chat_raw = st.text_area(
            "Enter chat messages (one per line)",
            value=("Watch out, they're flanking east!\n"
                   "I will ruin you, noob.\n"
                   "Nice shot! Let's push together."),
            height=100,
        )

        use_xgb = st.checkbox("Also show XGBoost prediction", value=False)
        submitted = st.form_submit_button("🔍 Analyze Player", use_container_width=True)

    if submitted:
        stats = {
            "assists": assists, "boosts": boosts, "damageDealt": damage,
            "DBNOs": dbnos, "headshotKills": headshotk, "heals": heals,
            "killPlace": kill_place, "killPoints": 1200, "kills": kills,
            "killStreaks": kill_stk, "longestKill": 87.0, "matchDuration": match_dur,
            "maxPlace": 99, "numGroups": num_groups, "rankPoints": 1500,
            "revives": revives, "rideDistance": ride_dist, "roadKills": 0,
            "swimDistance": swim_dist, "teamKills": 0, "vehicleDestroys": 0,
            "walkDistance": walk_dist, "weaponsAcquired": weapons, "winPoints": 1400,
        }
        chat_messages = [m.strip() for m in chat_raw.split("\n") if m.strip()]

        with st.spinner("Running PUBG Intelligence Pipeline..."):
            try:
                analyze_player = _get_pipeline()
                report = analyze_player(
                    player_id="dashboard_user",
                    gameplay_stats=stats,
                    chat_messages=chat_messages,
                    use_xgb=use_xgb,
                )
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                st.info("💡 If scaler is missing, it will be auto-generated on next startup.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")
                st.stop()

        st.success("✅ Analysis complete!")
        st.markdown("---")

        # ── Outcome prediction ─────────────────────────────────────────────
        col_pred, col_chart = st.columns([1, 2])
        with col_pred:
            st.markdown("### 🎮 Predicted Outcome")
            outcome = report.predicted_outcome
            badge_class = (
                "badge-win" if outcome == "Victory" else
                "badge-top10" if outcome == "Top-10 Finish" else "badge-elim"
            )
            st.markdown(
                f"<div class='metric-card'>"
                f"<span class='{badge_class}'>{outcome}</span><br><br>"
                f"<b>Confidence:</b> {report.outcome_probs[outcome]*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Toxicity Risk Score:** `{report.aggregate_toxicity:.3f}`")

        with col_chart:
            st.markdown("### Outcome Probability")
            labels = list(report.outcome_probs.keys())
            values = [v * 100 for v in report.outcome_probs.values()]
            colors = ["#da3633", "#388bfd", "#2ea043"]
            fig = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3"),
                yaxis=dict(title="Probability (%)", gridcolor="#30363d", range=[0, 105]),
                xaxis=dict(gridcolor="#30363d"),
                showlegend=False,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Feedback ───────────────────────────────────────────────────────
        st.markdown("### 💡 Tactical Feedback")
        for tip in report.feedback:
            css_type = (
                "danger"  if any(w in tip for w in ["⚠️", "🚨", "🔴"]) else
                "warning" if any(w in tip for w in ["💊", "🎯", "🤝"]) else
                "success" if "✅" in tip else ""
            )
            st.markdown(
                f"<div class='feedback-box {css_type}'>{tip}</div>",
                unsafe_allow_html=True,
            )

        # ── Chat toxicity breakdown ────────────────────────────────────────
        if report.toxicity_results:
            st.markdown("### 💬 Chat Toxicity Analysis")
            for result in report.toxicity_results:
                severity_color = {
                    "severe": "#da3633", "mild": "#e3b341", "clean": "#2ea043"
                }
                color = severity_color[result.severity]
                with st.expander(
                    f"{'🚨' if result.severity == 'severe' else '⚠️' if result.severity == 'mild' else '✅'}"
                    f" {result.message[:60]}{'...' if len(result.message) > 60 else ''}",
                ):
                    st.markdown(
                        f"**Severity:** <span style='color:{color}'>{result.severity.upper()}</span><br>"
                        f"**Flagged labels:** {', '.join(result.flagged_labels) if result.flagged_labels else 'None'}",
                        unsafe_allow_html=True,
                    )
                    score_df = pd.DataFrame({
                        "Label": list(result.scores.keys()),
                        "Score": list(result.scores.values()),
                    })
                    fig_tox = px.bar(
                        score_df, x="Score", y="Label", orientation="h",
                        color="Score", color_continuous_scale=["#2ea043", "#e3b341", "#da3633"],
                        range_color=[0, 1],
                    )
                    fig_tox.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e6edf3", size=12),
                        xaxis=dict(range=[0, 1], gridcolor="#30363d"),
                        yaxis=dict(gridcolor="#30363d"),
                        coloraxis_showscale=False,
                        margin=dict(t=10, b=10),
                        height=220,
                    )
                    st.plotly_chart(fig_tox, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Chat Moderation (bulk)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat Moderation":
    st.markdown("# 💬 Bulk Chat Moderation")
    st.markdown(
        "<p style='color:#8b949e'>Paste a batch of chat messages to detect toxic content.</p>",
        unsafe_allow_html=True,
    )

    chat_bulk = st.text_area(
        "Enter one message per line",
        value=(
            "Great shot!\n"
            "You are the worst player I have ever seen.\n"
            "Let's rotate to the safe zone.\n"
            "I will hack your account.\n"
            "Nice revive, thanks!\n"
            "Get out of this game you useless piece of garbage."
        ),
        height=200,
    )

    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)

    if st.button("🔍 Run Toxicity Detection", use_container_width=True):
        messages = [m.strip() for m in chat_bulk.split("\n") if m.strip()]

        with st.spinner("Running toxicity detection..."):
            try:
                predict_toxicity = _get_toxicity_fn()
                results = predict_toxicity(messages, threshold=threshold)
            except FileNotFoundError as e:
                st.error(f"❌ Model not found: {e}")
                st.stop()

        # Summary metrics
        n_toxic  = sum(1 for r in results if r.is_toxic)
        n_severe = sum(1 for r in results if r.severity == "severe")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", len(results))
        col2.metric("Toxic Messages", n_toxic, delta=f"{n_toxic/len(results)*100:.0f}%")
        col3.metric("Severe / Threats", n_severe)

        st.markdown("---")

        # Heatmap of all messages × labels
        st.markdown("### Toxicity Heatmap")
        heatmap_data = pd.DataFrame(
            [r.scores for r in results],
            index=[f"Msg {i+1}: {r.message[:30]}…" if len(r.message) > 30 else f"Msg {i+1}: {r.message}"
                   for i, r in enumerate(results)]
        )
        fig_heat = px.imshow(
            heatmap_data,
            color_continuous_scale=["#0d1117", "#e3b341", "#da3633"],
            zmin=0, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3", size=11),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Flagged messages table
        flagged = [(i+1, r.message, r.severity, ", ".join(r.flagged_labels))
                   for i, r in enumerate(results) if r.is_toxic]
        if flagged:
            st.markdown("### 🚨 Flagged Messages")
            flag_df = pd.DataFrame(flagged, columns=["#", "Message", "Severity", "Labels"])
            st.dataframe(flag_df, use_container_width=True)
        else:
            st.success("✅ No toxic messages detected at the current threshold.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Metrics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown("# 📊 Model Performance Metrics")
    METRICS_DIR = os.path.join(BASE_DIR, "metrics")

    col1, col2 = st.columns(2)

    # Outcome metrics
    with col1:
        st.markdown("### ⚔️ Outcome Predictor")
        outcome_metrics_path = os.path.join(METRICS_DIR, "outcome_metrics.json")
        if os.path.exists(outcome_metrics_path):
            with open(outcome_metrics_path) as f:
                om = json.load(f)
            for model_name, metrics in om.items():
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                c1, c2 = st.columns(2)
                c1.metric("Macro F1",  metrics.get("f1_macro",  "N/A"))
                c2.metric("Macro AUC", metrics.get("auc_macro", "N/A"))
        else:
            st.info("Run `python models/outcome_predictor.py` to generate metrics.")

        cm_path = os.path.join(METRICS_DIR, "confusion_tf_neural_network.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix — TF Neural Network")

        hist_path = os.path.join(METRICS_DIR, "outcome_training_history.png")
        if os.path.exists(hist_path):
            st.image(hist_path, caption="Outcome Predictor Training History")

    # Toxicity metrics
    with col2:
        st.markdown("### 💬 Toxicity Detector")
        # Check both metrics/ and saved_models/ (Colab saves there)
        tox_metrics_path = os.path.join(METRICS_DIR, "toxicity_metrics.json")
        if not os.path.exists(tox_metrics_path):
            tox_metrics_path = os.path.join(BASE_DIR, "saved_models", "toxicity_metrics.json")
        if os.path.exists(tox_metrics_path):
            with open(tox_metrics_path) as f:
                tm = json.load(f)
            c1, c2 = st.columns(2)
            c1.metric("Macro AUC", tm.get("macro_auc", "N/A"))
            c2.metric("Macro F1",  tm.get("macro_f1",  "N/A"))

            per_label_auc = tm.get("per_label_auc", {})
            if per_label_auc:
                fig = px.bar(
                    x=list(per_label_auc.keys()),
                    y=list(per_label_auc.values()),
                    labels={"x": "Label", "y": "AUC"},
                    color=list(per_label_auc.values()),
                    color_continuous_scale=["#e3b341", "#2ea043"],
                    range_color=[0.5, 1.0],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e6edf3"),
                    coloraxis_showscale=False,
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `python models/toxicity_detector.py` to generate metrics.")

        tox_hist = os.path.join(METRICS_DIR, "toxicity_training_history.png")
        if os.path.exists(tox_hist):
            st.image(tox_hist, caption="Toxicity Detector Training History")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Ethics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Ethics":
    st.markdown("# ⚖️ Ethics & Responsible AI")
    st.markdown("---")

    st.warning("""
    **False Positives in Toxicity Detection**

    No model is perfect. A phrase like *"I'll destroy you"* in competitive gaming context
    is friendly trash-talk, not a real threat. False positives risk silencing legitimate
    players and eroding trust in the system.

    **Mitigation:** Use confidence thresholds (≥0.7) before triggering sanctions,
    combined with human review for borderline cases.
    """)

    st.error("""
    **Bias in Training Data**

    The Jigsaw dataset reflects the demographic and linguistic biases of English-language
    internet comments. Minority dialects (AAVE, slang) may be over-flagged as toxic.
    Non-English chat is not modeled at all.

    **Mitigation:** Augment with diverse multilingual data; audit per-demographic F1 scores.
    """)

    st.info("""
    **Over-Censorship vs. Freedom of Expression**

    Heavy-handed moderation that mutes or bans players based on automated scoring alone
    creates a chilling effect on in-game communication, reducing squad coordination.

    **Mitigation:** Graduated response system — warning → mute → report. No automated
    permanent bans without human escalation.
    """)

    st.success("""
    **Feedback Loop Risk**

    If the outcome prediction model's feedback influences how players behave, and new
    match data collected from those players is used to retrain the model, the model
    may reinforce its own biases (e.g., always recommending passive play).

    **Mitigation:** Periodically retrain on randomized cohorts; monitor for behavioral
    homogenization across player segments.
    """)

    st.markdown("---")
    st.markdown("""
    ### Design Principles Applied
    | Principle | Implementation |
    |---|---|
    | Transparency | All model outputs include confidence scores |
    | Accountability | Every flag is logged with timestamp + model version |
    | Fairness | Per-demographic AUC auditing during evaluation |
    | Human oversight | Severe flags require human review before sanctions |
    | Right to appeal | Players can report false positives via in-game UI |
    """)
