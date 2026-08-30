import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from model_trainer import load_models
from utils import get_h2h, get_h2h_features, get_confidence, load_history, save_history, clear_history

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Football Predictor AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DARK MODE CSS
# ============================================================
def inject_css():
    st.markdown(
        """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .prediction-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #fafafa;
    }
    .prob-bar {
        background: #21262d;
        border-radius: 6px;
        height: 28px;
        margin: 4px 0;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 6px;
        display: flex;
        align-items: center;
        padding: 0 10px;
        color: #ffffff;
        font-weight: 600;
        font-size: 13px;
    }
    .confidence-high { color: #3fb950; font-weight: bold; }
    .confidence-med { color: #d29922; font-weight: bold; }
    .confidence-low { color: #f85149; font-weight: bold; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .shap-feature {
        display: flex;
        justify-content: space-between;
        padding: 6px 12px;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
    }
    .shap-positive { color: #3fb950; }
    .shap-negative { color: #f85149; }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #c9d1d9;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        border-color: #1f6feb !important;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

inject_css()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Settings")
    page = st.radio(
        "Navigate",
        ["Match Predictor", "League Table", "Prediction History"],
    )

# ============================================================
# LOAD MODELS & DATA
# ============================================================
models = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv("data/E0_Cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

df_raw = load_data()
teams_list = sorted(set(df_raw["HomeTeam"].unique()) | set(df_raw["AwayTeam"].unique()))

# ============================================================
# PAGE: MATCH PREDICTOR
# ============================================================
if page == "Match Predictor":
    st.title("Football Match Predictor")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Home Team")
        home_team = st.selectbox(
            "Home Team", teams_list,
            index=teams_list.index("Arsenal") if "Arsenal" in teams_list else 0,
            key="home_team",
        )
        h_goals = st.number_input("Avg Goals (Last 5)", 0.0, 5.0, 1.8, key="h_goals")
        h_shots = st.number_input("Avg Shots on Target", 0.0, 15.0, 5.5, key="h_shots")
        h_good = st.slider("Good Matches (0-5)", 0, 5, 3, key="h_good")

    with col2:
        st.markdown("### Away Team")
        away_team = st.selectbox(
            "Away Team", teams_list,
            index=teams_list.index("Liverpool") if "Liverpool" in teams_list else 0,
            key="away_team",
        )
        a_goals = st.number_input("Avg Goals (Last 5)", 0.0, 5.0, 1.0, key="a_goals")
        a_shots = st.number_input("Avg Shots on Target", 0.0, 15.0, 3.5, key="a_shots")
        a_good = st.slider("Good Matches (0-5)", 0, 5, 1, key="a_good")

    st.markdown("---")

    tab_result, tab_ou, tab_btts, tab_h2h = st.tabs(
        ["Match Result", "Over/Under 2.5", "Both Teams to Score", "Head-to-Head"]
    )

    if st.button("Predict Match", type="primary", use_container_width=True):
        h_off = 1 if (h_goals >= 1.6 or h_shots >= 5.0) else 0
        a_off = 1 if (a_goals >= 1.6 or a_shots >= 5.0) else 0
        h2h = get_h2h_features(df_raw, home_team, away_team)

        all_features = {
            "home_avg_goals": h_goals,
            "home_avg_shots": h_shots,
            "home_avg_conceded": a_goals,
            "home_good_matches": h_good,
            "home_is_offensive": h_off,
            "away_avg_goals": a_goals,
            "away_avg_shots": a_shots,
            "away_avg_conceded": h_goals,
            "away_good_matches": a_good,
            "away_is_offensive": a_off,
            "diff_avg_goals": h_goals - a_goals,
            "diff_avg_shots": h_shots - a_shots,
            "diff_avg_conceded": a_goals - h_goals,
            "diff_good_matches": h_good - a_good,
            "h2h_home_wr": h2h[0],
            "h2h_draw_wr": h2h[1],
            "h2h_away_wr": h2h[2],
        }

        # ---- RESULT TAB ----
        with tab_result:
            model_res, le_res, scaler_res, feat_names_res = models["result"]
            input_data = pd.DataFrame(
                [{col: all_features.get(col, 0) for col in feat_names_res}]
            )
            input_scaled = pd.DataFrame(
                scaler_res.transform(input_data), columns=feat_names_res
            )
            probs = model_res.predict_proba(input_scaled)[0]
            classes = le_res.classes_
            pred_idx = np.argmax(probs)
            pred_label = classes[pred_idx]
            conf_label, conf_class = get_confidence(probs)

            result_map = {}
            for i, c in enumerate(classes):
                if c == "H":
                    result_map["Home Win"] = probs[i]
                elif c == "D":
                    result_map["Draw"] = probs[i]
                elif c == "A":
                    result_map["Away Win"] = probs[i]

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader("Match Result")

            for label, prob in sorted(result_map.items(), key=lambda x: -x[1]):
                color = "#3fb950" if prob == max(result_map.values()) else "#1f6feb"
                pct = prob * 100
                st.markdown(
                    f"""
                <div class="prob-bar">
                    <div class="prob-fill" style="width: {pct}%; background: {color};">
                        {label}: {pct:.1f}%
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
            <div style="margin-top: 10px;">
                Confidence: <span class="{conf_class}">{conf_label}</span>
            </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("### Feature Importance")
            try:
                import shap
                explainer = (
                    shap.TreeExplainer(model_res.estimators_[1])
                    if hasattr(model_res, "estimators_")
                    else shap.LinearExplainer(model_res, input_scaled)
                )
                shap_values = explainer.shap_values(input_scaled)
                if isinstance(shap_values, list):
                    sv = shap_values[pred_idx]
                else:
                    sv = shap_values
                feature_impact = list(zip(feat_names_res, sv[0]))
                feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

                for fname, impact in feature_impact[:8]:
                    css_class = "shap-positive" if impact > 0 else "shap-negative"
                    sign = "+" if impact > 0 else "-"
                    st.markdown(
                        f"""
                    <div class="shap-feature">
                        <span>{fname}</span>
                        <span class="{css_class}">{sign} {abs(impact):.3f}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            except Exception:
                diffs = [
                    ("Goal Difference", h_goals - a_goals),
                    ("Shots Difference", h_shots - a_shots),
                    ("Good Matches Diff", h_good - a_good),
                    ("H2H Home Win Rate", h2h[0]),
                    ("H2H Draw Rate", h2h[1]),
                ]
                diffs.sort(key=lambda x: abs(x[1]), reverse=True)
                for fname, val in diffs:
                    css_class = "shap-positive" if val > 0 else "shap-negative"
                    sign = "+" if val > 0 else "-"
                    st.markdown(
                        f"""
                    <div class="shap-feature">
                        <span>{fname}</span>
                        <span class="{css_class}">{sign} {abs(val):.3f}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            save_history({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "home": home_team,
                "away": away_team,
                "prediction": pred_label,
                "prob_home": float(result_map.get("Home Win", 0)),
                "prob_draw": float(result_map.get("Draw", 0)),
                "prob_away": float(result_map.get("Away Win", 0)),
                "confidence": conf_label,
            })

        # ---- OVER/UNDER TAB ----
        with tab_ou:
            model_ou, scaler_ou, feat_names_ou = models["ou"]
            input_ou = pd.DataFrame(
                [{col: all_features.get(col, 0) for col in feat_names_ou}]
            )
            input_ou_scaled = pd.DataFrame(
                scaler_ou.transform(input_ou), columns=feat_names_ou
            )
            probs_ou = model_ou.predict_proba(input_ou_scaled)[0]
            prob_over = probs_ou[1] * 100 if len(probs_ou) > 1 else probs_ou[0] * 100
            prob_under = 100 - prob_over

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader("Over/Under 2.5 Goals")

            color_over = "#3fb950" if prob_over > prob_under else "#484f58"
            color_under = "#3fb950" if prob_under > prob_over else "#484f58"

            st.markdown(
                f"""
            <div class="prob-bar">
                <div class="prob-fill" style="width: {prob_over}%; background: {color_over};">
                    Over 2.5: {prob_over:.1f}%
                </div>
            </div>
            <div class="prob-bar">
                <div class="prob-fill" style="width: {prob_under}%; background: {color_under};">
                    Under 2.5: {prob_under:.1f}%
                </div>
            </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ---- BTTS TAB ----
        with tab_btts:
            model_btts, scaler_btts, feat_names_btts = models["btts"]
            input_btts = pd.DataFrame(
                [{col: all_features.get(col, 0) for col in feat_names_btts}]
            )
            input_btts_scaled = pd.DataFrame(
                scaler_btts.transform(input_btts), columns=feat_names_btts
            )
            probs_btts = model_btts.predict_proba(input_btts_scaled)[0]
            prob_yes = probs_btts[1] * 100 if len(probs_btts) > 1 else probs_btts[0] * 100
            prob_no = 100 - prob_yes

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader("Both Teams to Score")

            color_yes = "#3fb950" if prob_yes > prob_no else "#484f58"
            color_no = "#3fb950" if prob_no > prob_yes else "#484f58"

            st.markdown(
                f"""
            <div class="prob-bar">
                <div class="prob-fill" style="width: {prob_yes}%; background: {color_yes};">
                    Yes (Both Score): {prob_yes:.1f}%
                </div>
            </div>
            <div class="prob-bar">
                <div class="prob-fill" style="width: {prob_no}%; background: {color_no};">
                    No: {prob_no:.1f}%
                </div>
            </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ---- H2H TAB ----
        with tab_h2h:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader(f"{home_team} vs {away_team}")

            h2h_results = get_h2h(df_raw, home_team, away_team)
            if h2h_results:
                hw = sum(
                    1 for r in h2h_results
                    if r["result"] == "H" and r["home"] == home_team
                ) + sum(
                    1 for r in h2h_results
                    if r["result"] == "A" and r["home"] == away_team
                )
                dr = sum(1 for r in h2h_results if r["result"] == "D")
                aw = len(h2h_results) - hw - dr

                cols = st.columns(3)
                with cols[0]:
                    st.metric(f"{home_team} Wins", hw)
                with cols[1]:
                    st.metric("Draws", dr)
                with cols[2]:
                    st.metric(f"{away_team} Wins", aw)

                st.markdown("---")
                for r in h2h_results:
                    marker = (
                        "[W]"
                        if (r["result"] == "H" and r["home"] == home_team)
                        or (r["result"] == "A" and r["home"] == away_team)
                        else "[L]" if r["result"] != "D" else "[D]"
                    )
                    st.markdown(
                        f"**{r['date']}**: {r['home']} {r['score']} {r['away']} {marker}"
                    )
            else:
                st.info("No head-to-head records found.")

            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PAGE: LEAGUE TABLE
# ============================================================
elif page == "League Table":
    st.title("League Table")

    if os.path.exists("data/league_table.csv"):
        df_league = pd.read_csv("data/league_table.csv")
    else:
        df_league = pd.DataFrame()

    if not df_league.empty:
        st.dataframe(
            df_league.style.apply(
                lambda row: [
                    "background-color: #1a3a2a" if row["Position"] <= 4
                    else "background-color: #3a1a1a" if row["Position"] >= 18
                    else "background-color: transparent"
                    for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("**Top 4** - Champions League spots")
        st.markdown("**Bottom 3** - Relegation zone")
    else:
        st.warning("League table not available.")


# ============================================================
# PAGE: PREDICTION HISTORY
# ============================================================
elif page == "Prediction History":
    st.title("Prediction History")

    history = load_history()

    if not history:
        st.info("No predictions yet. Make your first prediction!")
    else:
        df_hist = pd.DataFrame(history)

        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Predictions", len(df_hist))
        with cols[1]:
            if "prediction" in df_hist.columns:
                most_common = df_hist["prediction"].mode()[0] if len(df_hist) > 0 else "N/A"
                st.metric("Most Predicted", most_common)
        with cols[2]:
            if "confidence" in df_hist.columns:
                high_conf = (df_hist["confidence"] == "HIGH").sum()
                st.metric("High Confidence", high_conf)

        st.markdown("---")

        filter_team = st.selectbox("Filter by Team", ["All"] + teams_list)
        if filter_team != "All":
            df_hist = df_hist[
                (df_hist["home"] == filter_team) | (df_hist["away"] == filter_team)
            ]

        for _, row in df_hist.head(20).iterrows():
            pred_label = row.get("prediction", "?")
            pred_text = {"H": "Home Win", "D": "Draw", "A": "Away Win"}.get(pred_label, pred_label)
            st.markdown(
                f"""
            <div class="prediction-card" style="padding: 12px;">
                <div style="display: flex; justify-content: space-between;">
                    <span><strong>{row.get('home', '?')}</strong> vs <strong>{row.get('away', '?')}</strong></span>
                    <span>{pred_text}</span>
                </div>
                <div style="font-size: 12px; color: #8b949e; margin-top: 5px;">
                    {row.get('date', 'N/A')} | H:{row.get('prob_home', 0)*100:.0f}% D:{row.get('prob_draw', 0)*100:.0f}% A:{row.get('prob_away', 0)*100:.0f}%
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if st.button("Clear History"):
            clear_history()
            st.rerun()
