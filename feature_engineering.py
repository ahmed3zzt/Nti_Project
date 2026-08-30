import pandas as pd
import numpy as np


def clean_raw_data(input_path="data/data.csv", output_path="data/E0_Cleaned.csv"):
    """Step 1: Clean raw match data by selecting relevant columns."""
    df = pd.read_csv(input_path)

    base_columns = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST",
    ]

    all_cols_needed = [c for c in base_columns if c in df.columns]
    df_clean = df[all_cols_needed].copy()
    df_clean = df_clean.dropna(subset=["FTHG", "FTAG", "FTR"])
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], dayfirst=True)
    df_clean = df_clean.sort_values("Date").reset_index(drop=True)
    df_clean.to_csv(output_path, index=False)
    return df_clean


def compute_last5_stats(df_all, team, match_date):
    """Compute rolling last-5-match statistics for a team before a given date.

    Returns:
        tuple: (avg_goals, avg_shots_on_target, avg_goals_conceded,
                good_matches_count, is_offensive_flag) or None if < 5 matches.
    """
    past = df_all[
        ((df_all["HomeTeam"] == team) | (df_all["AwayTeam"] == team))
        & (df_all["Date"] < match_date)
    ].sort_values("Date").tail(5)

    if len(past) < 5:
        return None

    goals_scored = []
    shots_on_target = []
    goals_conceded = []
    good_matches = 0

    for _, row in past.iterrows():
        is_home = row["HomeTeam"] == team
        g_for = row["FTHG"] if is_home else row["FTAG"]
        g_against = row["FTAG"] if is_home else row["FTHG"]
        shots = row["HST"] if is_home else row["AST"]
        win = (is_home and row["FTR"] == "H") or (not is_home and row["FTR"] == "A")

        goals_scored.append(g_for)
        shots_on_target.append(shots)
        goals_conceded.append(g_against)
        if win or g_for >= 2 or shots >= 6:
            good_matches += 1

    avg_g = np.mean(goals_scored)
    avg_s = np.mean(shots_on_target)
    avg_gc = np.mean(goals_conceded)
    is_offensive = 1 if (avg_g >= 1.6 or avg_s >= 5.0) else 0

    return avg_g, avg_s, avg_gc, good_matches, is_offensive


def compute_h2h_features(df_all, home_team, away_team, match_date):
    """Compute head-to-head win/draw/away rates between two teams.

    Returns:
        tuple: (home_win_rate, draw_rate, away_win_rate)
    """
    past = df_all[df_all["Date"] < match_date].copy()
    h2h = past[
        ((past["HomeTeam"] == home_team) & (past["AwayTeam"] == away_team))
        | ((past["HomeTeam"] == away_team) & (past["AwayTeam"] == home_team))
    ].tail(10)

    if len(h2h) == 0:
        return 0, 0, 0

    home_wins = 0
    draws = 0
    away_wins = 0
    for _, row in h2h.iterrows():
        if row["FTR"] == "D":
            draws += 1
        elif (row["HomeTeam"] == home_team and row["FTR"] == "H") or (
            row["AwayTeam"] == home_team and row["FTR"] == "A"
        ):
            home_wins += 1
        else:
            away_wins += 1

    total = len(h2h)
    return home_wins / total, draws / total, away_wins / total


def build_feature_dataset(df_clean):
    """Step 2: Build the full feature dataset from cleaned match data.

    For each match, computes:
        - Last-5-match rolling stats (goals, shots, conceded, good matches, offensive flag)
        - Head-to-head rates
        - Difference features between home and away team stats
        - Target (match result: H/D/A)

    Returns:
        pd.DataFrame: Feature-engineered dataset.
    """
    features_list = []

    for _, row in df_clean.iterrows():
        home_stats = compute_last5_stats(df_clean, row["HomeTeam"], row["Date"])
        away_stats = compute_last5_stats(df_clean, row["AwayTeam"], row["Date"])

        if home_stats is not None and away_stats is not None:
            h2h = compute_h2h_features(
                df_clean, row["HomeTeam"], row["AwayTeam"], row["Date"]
            )

            feat = {
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "Date": row["Date"],
                "home_avg_goals": home_stats[0],
                "home_avg_shots": home_stats[1],
                "home_avg_conceded": home_stats[2],
                "home_good_matches": home_stats[3],
                "home_is_offensive": home_stats[4],
                "away_avg_goals": away_stats[0],
                "away_avg_shots": away_stats[1],
                "away_avg_conceded": away_stats[2],
                "away_good_matches": away_stats[3],
                "away_is_offensive": away_stats[4],
                "diff_avg_goals": home_stats[0] - away_stats[0],
                "diff_avg_shots": home_stats[1] - away_stats[1],
                "diff_avg_conceded": home_stats[2] - away_stats[2],
                "diff_good_matches": home_stats[3] - away_stats[3],
                "h2h_home_wr": h2h[0],
                "h2h_draw_wr": h2h[1],
                "h2h_away_wr": h2h[2],
                "Target": row["FTR"],
            }
            features_list.append(feat)

    df_feat = pd.DataFrame(features_list)
    df_feat.to_csv("data/dataset_features.csv", index=False)
    return df_feat


def prepare_result_features(df_feat):
    """Prepare feature matrix and labels for the match result model.

    Returns:
        tuple: (X_scaled, y, label_encoder, scaler, feature_names, df_model)
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    feature_cols = [c for c in df_feat.columns if c not in ["Target", "HomeTeam", "AwayTeam", "Date"]]
    df_model = df_feat.dropna(subset=[c for c in feature_cols if c in df_feat.columns], how="any").copy()

    le = LabelEncoder()
    y = le.fit_transform(df_model["Target"])
    X = df_model.drop(columns=["Target", "HomeTeam", "AwayTeam", "Date"], errors="ignore")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, le, scaler, list(X.columns), df_model


def prepare_ou_features(df_feat, df_clean):
    """Prepare feature matrix and labels for the Over/Under 2.5 model.

    Returns:
        tuple: (X_scaled, y, scaler, feature_names)
    """
    from sklearn.preprocessing import StandardScaler

    goals_map = {}
    for _, row in df_clean.iterrows():
        key = (row["HomeTeam"], row["AwayTeam"], str(row["Date"].date()))
        goals_map[key] = row["FTHG"] + row["FTAG"]

    df_feat = df_feat.copy()
    df_feat["total_goals"] = df_feat.apply(
        lambda r: goals_map.get(
            (r["HomeTeam"], r["AwayTeam"],
             str(r["Date"].date()) if hasattr(r["Date"], "date") else str(r["Date"])[:10]),
            None,
        ),
        axis=1,
    )

    df_ou = df_feat.dropna(subset=["total_goals"]).copy()
    df_ou["over_2_5"] = (df_ou["total_goals"] > 2.5).astype(int)

    feature_cols = [c for c in df_ou.columns if c not in ["Target", "total_goals", "over_2_5", "HomeTeam", "AwayTeam", "Date"]]
    df_ou_model = df_ou.dropna(subset=feature_cols)
    X = df_ou_model[feature_cols]
    y = df_ou_model["over_2_5"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, scaler, list(X.columns)


def prepare_btts_features(df_feat, df_clean):
    """Prepare feature matrix and labels for the BTTS model.

    Returns:
        tuple: (X_scaled, y, scaler, feature_names)
    """
    from sklearn.preprocessing import StandardScaler

    btts_data = []
    for _, row in df_clean.iterrows():
        btts_data.append({
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "Date": row["Date"],
            "both_scored": 1 if (row["FTHG"] > 0 and row["FTAG"] > 0) else 0,
        })
    df_btts = pd.DataFrame(btts_data)

    btts_features = []
    for _, row in df_feat.iterrows():
        match = df_btts[
            (df_btts["HomeTeam"] == row["HomeTeam"])
            & (df_btts["AwayTeam"] == row["AwayTeam"])
            & (df_btts["Date"] == row["Date"])
        ]
        if len(match) > 0:
            feat = row.to_dict()
            feat["both_scored"] = match.iloc[0]["both_scored"]
            btts_features.append(feat)

    df_btts_all = pd.DataFrame(btts_features)
    feature_cols = [c for c in df_btts_all.columns if c not in ["Target", "total_goals", "both_scored", "HomeTeam", "AwayTeam", "Date"]]
    df_btts_model = df_btts_all.dropna(subset=feature_cols)
    X = df_btts_model[feature_cols]
    y = df_btts_model["both_scored"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, scaler, list(X.columns)
