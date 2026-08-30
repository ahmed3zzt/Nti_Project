# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

from feature_engineering import (
    clean_raw_data,
    build_feature_dataset,
    prepare_result_features,
    prepare_ou_features,
    prepare_btts_features,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False
    print("XGBoost not available. Install with: pip install xgboost")


# ============================================================
# STEP 1: Clean raw data
# ============================================================
print("Step 1: Cleaning raw data...")
df_clean = clean_raw_data()
print(f"  Saved: data/E0_Cleaned.csv ({len(df_clean)} rows)")


# ============================================================
# STEP 2: Feature Engineering
# ============================================================
print("\nStep 2: Feature engineering...")
df_feat = build_feature_dataset(df_clean)
print(f"  Extracted {len(df_feat)} samples with {len(df_feat.columns)} features")


# ============================================================
# STEP 3: Train Match Result Model
# ============================================================
print("\n" + "=" * 50)
print("TRAINING MATCH RESULT MODEL")
print("=" * 50)

X_scaled, y, le, scaler, feature_names, df_model = prepare_result_features(df_feat)
print(f"  Features: {len(feature_names)}")
print(f"  Samples: {len(X_scaled)}")

models_dict = {
    'LR': LogisticRegression(max_iter=2000, random_state=42),
    'RF': RandomForestClassifier(n_estimators=200, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
}
if HAS_XGB:
    models_dict['XGB'] = XGBClassifier(
        n_estimators=200, learning_rate=0.1, use_label_encoder=False,
        eval_metric='mlogloss', random_state=42, verbosity=0
    )

print("\n  Cross-validation (5-fold):")
for name, model in models_dict.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"    {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

estimators = [(name, model) for name, model in models_dict.items()]
ensemble = VotingClassifier(
    estimators=estimators, voting='soft',
    weights=[1, 2, 2, 2][:len(models_dict)]
)
ensemble.fit(X_scaled, y)

cv_ens = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ens_scores = cross_val_score(ensemble, X_scaled, y, cv=cv_ens, scoring='accuracy')
print(f"    Ensemble: {ens_scores.mean():.4f} (+/- {ens_scores.std():.4f})")

joblib.dump(ensemble, os.path.join(MODEL_DIR, 'best_match_predictor_model.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder_result.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_result.pkl'))
joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names_result.pkl'))
print("  Saved: models/best_match_predictor_model.pkl, models/label_encoder_result.pkl, models/scaler_result.pkl, models/feature_names_result.pkl")


# ============================================================
# STEP 4: Train Over/Under 2.5 Model
# ============================================================
print("\n" + "=" * 50)
print("TRAINING OVER/UNDER 2.5 MODEL")
print("=" * 50)

X_ou, y_ou, scaler_ou, feat_names_ou = prepare_ou_features(df_feat, df_clean)
print(f"  Features: {len(feat_names_ou)}")
print(f"  Samples: {len(X_ou)}")

ou_models = {
    'RF': RandomForestClassifier(n_estimators=200, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
}
if HAS_XGB:
    ou_models['XGB'] = XGBClassifier(
        n_estimators=200, learning_rate=0.1, use_label_encoder=False,
        eval_metric='logloss', random_state=42, verbosity=0
    )

print("\n  Cross-validation (5-fold):")
for name, model in ou_models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_ou, y_ou, cv=cv, scoring='accuracy')
    print(f"    {name}: {scores.mean():.4f}")

ou_estimators = [(name, model) for name, model in ou_models.items()]
ou_ensemble = VotingClassifier(estimators=ou_estimators, voting='soft')
ou_ensemble.fit(X_ou, y_ou)

joblib.dump(ou_ensemble, os.path.join(MODEL_DIR, 'over_under_model.pkl'))
joblib.dump(scaler_ou, os.path.join(MODEL_DIR, 'scaler_ou.pkl'))
joblib.dump(feat_names_ou, os.path.join(MODEL_DIR, 'feature_names_ou.pkl'))
print("  Saved: models/over_under_model.pkl, models/scaler_ou.pkl, models/feature_names_ou.pkl")


# ============================================================
# STEP 5: Train BTTS Model
# ============================================================
print("\n" + "=" * 50)
print("TRAINING BTTS MODEL")
print("=" * 50)

X_btts, y_btts, scaler_btts, feat_names_btts = prepare_btts_features(df_feat, df_clean)
print(f"  Features: {len(feat_names_btts)}")
print(f"  Samples: {len(X_btts)}")

btts_models = {
    'RF': RandomForestClassifier(n_estimators=200, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
}
if HAS_XGB:
    btts_models['XGB'] = XGBClassifier(
        n_estimators=200, learning_rate=0.1, use_label_encoder=False,
        eval_metric='logloss', random_state=42, verbosity=0
    )

btts_estimators = [(name, model) for name, model in btts_models.items()]
btts_ensemble = VotingClassifier(estimators=btts_estimators, voting='soft')
btts_ensemble.fit(X_btts, y_btts)

cv_btts = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
btts_scores = cross_val_score(btts_ensemble, X_btts, y_btts, cv=cv_btts, scoring='accuracy')
print(f"  BTTS Ensemble: {btts_scores.mean():.4f}")

joblib.dump(btts_ensemble, os.path.join(MODEL_DIR, 'btts_model.pkl'))
joblib.dump(scaler_btts, os.path.join(MODEL_DIR, 'scaler_btts.pkl'))
joblib.dump(feat_names_btts, os.path.join(MODEL_DIR, 'feature_names_btts.pkl'))
print("  Saved: models/btts_model.pkl, models/scaler_btts.pkl, models/feature_names_btts.pkl")


# ============================================================
# STEP 6: Build League Table
# ============================================================
print("\n" + "=" * 50)
print("BUILDING LEAGUE TABLE")
print("=" * 50)

df_goals = pd.read_csv('data/E0_Cleaned.csv')
df_goals['Date'] = pd.to_datetime(df_goals['Date'], dayfirst=True)
df_goals = df_goals.sort_values('Date').reset_index(drop=True)

teams = set(df_goals['HomeTeam'].unique()) | set(df_goals['AwayTeam'].unique())
league = {t: {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0} for t in teams}

for _, row in df_goals.iterrows():
    ht, at = row['HomeTeam'], row['AwayTeam']
    hg, ag = row['FTHG'], row['FTAG']
    league[ht]['P'] += 1
    league[at]['P'] += 1
    league[ht]['GF'] += hg
    league[ht]['GA'] += ag
    league[at]['GF'] += ag
    league[at]['GA'] += hg
    if row['FTR'] == 'H':
        league[ht]['W'] += 1
        league[ht]['Pts'] += 3
        league[at]['L'] += 1
    elif row['FTR'] == 'A':
        league[at]['W'] += 1
        league[at]['Pts'] += 3
        league[ht]['L'] += 1
    else:
        league[ht]['D'] += 1
        league[at]['D'] += 1
        league[ht]['Pts'] += 1
        league[at]['Pts'] += 1
    league[ht]['GD'] = league[ht]['GF'] - league[ht]['GA']
    league[at]['GD'] = league[at]['GF'] - league[at]['GA']

df_league = pd.DataFrame(league).T
df_league.index.name = 'Team'
df_league = df_league.sort_values(['Pts', 'GD', 'GF'], ascending=False).reset_index()
df_league['Position'] = range(1, len(df_league) + 1)
df_league = df_league[['Position', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']]
df_league.to_csv('data/league_table.csv', index=False)
print(df_league.to_string(index=False))
print("\n  Saved: data/league_table.csv")

print("\n" + "=" * 50)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 50)
