# Machine Learning Models Guide

This document explains the machine learning models used in the Football Match Predictor project, why each was chosen, and how they work together.

---

## Project Overview

The system uses **three separate ensemble models** to predict different aspects of a football match:

1. **Match Result** - Predicts Home Win, Draw, or Away Win
2. **Over/Under 2.5 Goals** - Predicts whether total goals will be over or under 2.5
3. **Both Teams to Score (BTTS)** - Predicts whether both teams will score at least one goal

All three models use **VotingClassifier** ensembles that combine multiple base learners for more robust predictions.

---

## Models Used

### 1. Logistic Regression (LR)

**What it is:** A linear model that estimates the probability of each class using a logistic (sigmoid) function applied to a linear combination of input features.

**Why it is used here:**
- Provides a fast, interpretable baseline
- Works well when the relationship between features and target is approximately linear
- Regularized variants (L2) prevent overfitting on small datasets
- Serves as a "sanity check" - if LR performs well, the problem has strong linear signals

**Role in the ensemble:** Included in the Match Result model only. Its predictions help anchor the ensemble, preventing the more complex models from overfitting to noise.

**Key parameters:**
- `max_iter=2000` - Allows enough iterations for convergence on this feature set
- Uses L2 regularization by default (C=1.0)

---

### 2. Random Forest (RF)

**What it is:** An ensemble of decision trees, each trained on a random subset of the data and features. Final prediction is the majority vote (classification) of all trees.

**Why it is used here:**
- Handles non-linear relationships between features naturally
- Resistant to overfitting due to bagging (bootstrap aggregation) and feature randomization
- Works well with mixed feature types (continuous stats, binary flags, rates)
- Provides feature importance rankings for interpretability

**Role in the ensemble:** Core model in all three prediction tasks. Its robustness and ability to capture complex interactions (e.g., how offensive strength interacts with H2H history) makes it the primary workhorse.

**Key parameters:**
- `n_estimators=200` - 200 trees provide stable predictions
- `random_state=42` - Reproducible results

---

### 3. Gradient Boosting (GB)

**What it is:** An ensemble of weak decision trees built sequentially, where each new tree corrects the errors of the previous ones. Final prediction is the weighted sum of all trees.

**Why it is used here:**
- Often achieves the highest single-model accuracy on structured/tabular data
- Focuses on hard-to-classify samples by learning from residual errors
- Handles feature interactions and non-linearities effectively

**Role in the ensemble:** Typically the strongest individual performer. Its sequential error-correction complements the parallel approach of Random Forest.

**Key parameters:**
- `n_estimators=200` - Number of boosting stages
- `learning_rate=0.1` - Step size shrinkage to prevent overfitting

---

### 4. XGBoost (XGB) - Optional

**What it is:** An optimized, highly efficient implementation of gradient boosting with regularization, parallel processing, and handling of missing values.

**Why it is used here:**
- State-of-the-art performance on structured data
- Built-in L1/L2 regularization reduces overfitting
- Handles missing values natively (useful when some features are unavailable)
- Faster training than standard Gradient Boosting

**Role in the ensemble:** When available, it replaces or supplements the standard Gradient Boosting model. Often the highest-performing individual model.

**Key parameters:**
- `n_estimators=200, learning_rate=0.1`
- `eval_metric='mlogloss'` (multiclass) or `'logloss'` (binary)
- `use_label_encoder=False` - Uses the project's own LabelEncoder
- `verbosity=0` - Suppresses training output

---

## Ensemble Strategy: VotingClassifier

### How it works

The **VotingClassifier** combines predictions from multiple models using **soft voting** - it averages the predicted probabilities from each model rather than just counting votes.

```
Final probability = weighted average of all model probabilities
```

### Why soft voting over hard voting

- **Soft voting** considers the confidence (probability) of each prediction, not just the class label
- A model that is 90% confident in "Home Win" contributes more influence than one that is 51% confident
- Produces smoother, more calibrated probability estimates

### Weighting

The Match Result model uses weights `[1, 2, 2, 2]` for `[LR, RF, GB, XGB]`:
- Logistic Regression gets weight 1 (lower influence)
- Tree-based models get weight 2 (higher influence)
- This reflects that tree-based models typically capture the non-linear patterns in football data better

The O/U and BTTS models use equal weights (no explicit weighting).

---

## Feature Set

All models share the same core features (16 base features):

| Feature | Description |
|---------|-------------|
| `home_avg_goals` | Home team's average goals scored in last 5 matches |
| `home_avg_shots` | Home team's average shots on target in last 5 matches |
| `home_avg_conceded` | Home team's average goals conceded in last 5 matches |
| `home_good_matches` | Number of "good" performances in last 5 (win, 2+ goals, or 6+ shots) |
| `home_is_offensive` | Binary flag: 1 if avg goals >= 1.6 or avg shots >= 5.0 |
| `away_avg_goals` | Same as above but for away team |
| `away_avg_shots` | Same as above but for away team |
| `away_avg_conceded` | Same as above but for away team |
| `away_good_matches` | Same as above but for away team |
| `away_is_offensive` | Same as above but for away team |
| `diff_avg_goals` | Home goals minus away goals (positive = home stronger) |
| `diff_avg_shots` | Home shots minus away shots |
| `diff_avg_conceded` | Home conceded minus away conceded |
| `diff_good_matches` | Home good matches minus away good matches |
| `h2h_home_wr` | Head-to-head home win rate (last 10 meetings) |
| `h2h_draw_wr` | Head-to-head draw rate |
| `h2h_away_wr` | Head-to-head away win rate |

**Note:** The feature set is built entirely from match statistics (goals, shots, head-to-head records). No external market data is used.

---

## Why an Ensemble of Ensembles?

The three prediction tasks (Result, O/U, BTTS) are **different classification problems** with different label distributions:

- **Result**: 3-class (H/D/A) - imbalanced, home wins are most common
- **Over/Under 2.5**: Binary - roughly 45-55 split
- **BTTS**: Binary - roughly 50-50 split

Using separate models for each task allows:
1. Task-specific feature selection and optimization
2. Independent hyperparameter tuning
3. Better calibration for each problem's difficulty

The soft-voting ensemble within each task reduces variance and improves generalization compared to any single model.

---

## Model Persistence

All trained artifacts are saved as `.pkl` files using `joblib`:

| File | Contents |
|------|----------|
| `best_match_predictor_model.pkl` | Match Result VotingClassifier ensemble |
| `label_encoder_result.pkl` | LabelEncoder mapping H/D/A to integers |
| `scaler_result.pkl` | StandardScaler for result model features |
| `feature_names_result.pkl` | List of feature column names for result model |
| `over_under_model.pkl` | Over/Under 2.5 VotingClassifier ensemble |
| `scaler_ou.pkl` | StandardScaler for O/U features |
| `feature_names_ou.pkl` | Feature names for O/U model |
| `btts_model.pkl` | BTTS VotingClassifier ensemble |
| `scaler_btts.pkl` | StandardScaler for BTTS features |
| `feature_names_btts.pkl` | Feature names for BTTS model |

---

## Training Pipeline

```
data/data.csv
  --> clean_raw_data() --> data/E0_Cleaned.csv
  --> build_feature_dataset() --> data/dataset_features.csv
  --> prepare_result_features() --> model training
  --> prepare_ou_features() --> model training
  --> prepare_btts_features() --> model training
  --> data/league_table.csv
```

Each training step:
1. Prepares features and labels
2. Runs 5-fold stratified cross-validation for each base model
3. Prints CV accuracy scores
4. Trains the final VotingClassifier on the full dataset
5. Saves the model and associated artifacts

---

## Interpreting Predictions

### SHAP (SHapley Additive exPlanations)

The app uses SHAP to explain individual predictions. SHAP values show how much each feature contributes to pushing the prediction away from the baseline (average) prediction:

- **Positive SHAP value** (+): Feature pushes toward the predicted class
- **Negative SHAP value** (-): Feature pushes away from the predicted class
- **Magnitude**: Larger absolute values indicate stronger influence

### Confidence Levels

Prediction confidence is based on the maximum class probability:

| Level | Threshold | Meaning |
|-------|-----------|---------|
| HIGH | >= 55% | Strong prediction, high agreement among models |
| MEDIUM | 40-55% | Moderate confidence, some disagreement |
| LOW | < 40% | Uncertain, models disagree significantly |
