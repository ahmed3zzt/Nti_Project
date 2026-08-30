import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


HISTORY_FILE = "data/prediction_history.json"


def get_h2h(df_raw, home_team, away_team, n=10):
    """Get head-to-head match results between two teams."""
    past = df_raw[df_raw["Date"] < df_raw["Date"].max()].copy()
    h2h = past[
        ((past["HomeTeam"] == home_team) & (past["AwayTeam"] == away_team))
        | ((past["HomeTeam"] == away_team) & (past["AwayTeam"] == home_team))
    ].tail(n)

    results = []
    for _, row in h2h.iterrows():
        results.append({
            "date": str(row["Date"].date()),
            "home": row["HomeTeam"],
            "away": row["AwayTeam"],
            "score": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            "result": row["FTR"],
        })
    return results


def get_h2h_features(df_raw, home_team, away_team):
    """Get head-to-head win/draw/away rates for prediction features."""
    past = df_raw[df_raw["Date"] < df_raw["Date"].max()].copy()
    h2h = past[
        ((past["HomeTeam"] == home_team) & (past["AwayTeam"] == away_team))
        | ((past["HomeTeam"] == away_team) & (past["AwayTeam"] == home_team))
    ].tail(10)

    if len(h2h) == 0:
        return 0, 0, 0

    hw, dr, aw = 0, 0, 0
    for _, row in h2h.iterrows():
        if row["FTR"] == "D":
            dr += 1
        elif (row["HomeTeam"] == home_team and row["FTR"] == "H") or (
            row["AwayTeam"] == home_team and row["FTR"] == "A"
        ):
            hw += 1
        else:
            aw += 1

    total = len(h2h)
    return hw / total, dr / total, aw / total


def get_confidence(probs):
    """Classify prediction confidence based on max probability."""
    max_p = max(probs)
    if max_p >= 0.55:
        return "HIGH", "confidence-high"
    elif max_p >= 0.40:
        return "MEDIUM", "confidence-med"
    else:
        return "LOW", "confidence-low"


def load_history():
    """Load prediction history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(entry):
    """Save a prediction entry to history (newest first)."""
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def clear_history():
    """Clear all prediction history."""
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
