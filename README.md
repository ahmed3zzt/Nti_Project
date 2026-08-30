# Football Predictor AI

A modern football (soccer) match prediction web application built with Streamlit. Predict match outcomes, over/under 2.5 goals, and both teams to score using machine learning models trained on historical Premier League data.

## 🚀 Live Demo

![Streamlit App](https://static.streamlit.io/badges/streamlit-badge-dark.png)

## 📋 Overview

This project implements three prediction models for English Premier League matches:

| Prediction Type | Model | Description |
|----------------|-------|-------------|
| **Match Result** (Home Win / Draw / Away Win) | Gradient Boosting Classifier | Predicts the most likely match outcome based on team statistics and head-to-head history |
| **Over/Under 2.5 Goals** | Gradient Boosting Classifier | Predicts whether total goals will be over or under 2.5 |
| **Both Teams to Score (BTTS)** | Gradient Boosting Classifier | Predicts if both teams will score at least one goal |

## 🛠️ Technology Stack

- **Language**: Python 3.x
- **Framework**: Streamlit (for interactive web UI)
- **ML Libraries**: scikit-learn, xgboost, lightgbm
- **Data Visualization**: SHAP (for feature importance), pandas, numpy
- **Model Persistence**: joblib (pickle files)
- **CSS**: Custom dark theme with modern UI components

## 📊 Project Structure

```
football-predictor-ai/
├── app.py                  # Main Streamlit application
├── train.py                # Training script
├── model_trainer.py        # Model training and loading logic
├── utils.py                # Utility functions (history, H2H, confidence)
├── feature_engineering.py  # Feature engineering & dataset creation
├── data/
│   ├── E0_Cleaned.csv      # Raw match data (2025-2026 season)
│   ├── data.csv            # Full dataset with betting odds
│   ├── dataset_features.csv # Engineered features for training
│   ├── league_table.csv    # Current season standings
│   └── prediction_history.json # Saved predictions
├── models/
│   ├── best_match_predictor_model.pkl  # Result model
│   ├── over_under_model.pkl            # Over/Under model
│   ├── btts_model.pkl                  # BTTS model
│   ├── scaler_result.pkl               # Result scaler
│   ├── scaler_ou.pkl                   # Over/Under scaler
│   ├── scaler_btts.pkl                 # BTTS scaler
│   ├── feature_names_result.pkl        # Result feature names
│   ├── feature_names_ou.pkl            # O/U feature names
│   ├── feature_names_btts.pkl          # BTTS feature names
│   ├── label_encoder_result.pkl        # Label encoder for result classes
│   └── MODELS_GUIDE.md                 # Detailed model guide
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🤖 How Models Are Trained

### Training Pipeline (`train.py`)

1. **Data Loading**: Loads `data/dataset_features.csv` containing engineered features from historical matches
2. **Feature Selection**: Uses 19 features including:
   - Average goals scored/conceded (home/away)
   - Average shots on target (home/away)
   - "Good matches" count (last 5 games)
   - Offensive indicator (goals/shots thresholds)
   - Head-to-head win/draw/loss rates
   - Goal/shot differentials
3. **Train-Test Split**: 80/20 split with random state for reproducibility
4. **Model Training**: Gradient Boosting Classifier for each prediction type
5. **Scaling**: StandardScaler fitted on training data
6. **Evaluation**: Accuracy and classification report on test set
7. **Saving**: Trained models, scalers, feature names, and label encoders saved as `.pkl` files

### Feature Engineering (`feature_engineering.py`)

- Computes rolling averages over last 5 matches
- Calculates head-to-head statistics (home win/draw/away win rates)
- Creates difference features (home vs away team stats)
- Encodes categorical variables (team names)
- Target encoding for match result, over/under, and BTTS

## 🔮 How to Use the App

### 1. Launch the Application

```bash
streamlit run app.py
```

### 2. Prediction Tabs

The app has four main tabs accessible via sidebar navigation:

#### Match Result
- Select home and away teams
- Input statistical averages (goals, shots, good matches)
- Click **Predict Match Result**
- View probability bars for Home Win, Draw, Away Win
- SHAP feature importance chart showing which factors influenced the prediction

#### Over/Under 2.5 Goals
- Uses same input features as match result
- Predicts probability of over 2.5 goals vs under 2.5 goals

#### Both Teams to Score
- Predicts if both teams will score (Yes/No)
- Uses selected team statistics to determine likelihood

#### Head-to-Head
- View historical results between selected teams
- Show win/draw/loss counts and recent match history

### 3. Saving Predictions

- Each prediction is automatically saved to `data/prediction_history.json`
- View prediction history in the sidebar under "Prediction History"
- Filter by team, see most predicted outcomes, and high-confidence predictions

## 🏗️ Model Architecture

Each model uses **Gradient Boosting Classification** with the following configuration:

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

**Input Features (19 dimensions)**:
| Feature | Description |
|---------|-------------|
| `home_avg_goals` | Home team's average goals per game (last 5) |
| `home_avg_shots` | Home team's average shots on target |
| `home_avg_conceded` | Home team's average goals conceded |
| `home_good_matches` | Home team's "good" matches count |
| `home_is_offensive` | Binary: goals >= 1.6 or shots >= 5.0 |
| `away_avg_goals` | Away team's average goals per game |
| `away_avg_shots` | Away team's average shots on target |
| `away_avg_conceded` | Away team's average goals conceded |
| `away_good_matches` | Away team's "good" matches count |
| `away_is_offensive` | Binary: goals >= 1.6 or shots >= 5.0 |
| `diff_avg_goals` | Goal difference (home - away) |
| `diff_avg_shots` | Shot difference (home - away) |
| `diff_avg_conceded` | Conceded difference (away - home) |
| `diff_good_matches` | Good matches difference |
| `h2h_home_wr` | Head-to-head home win rate |
| `h2h_draw_wr` | Head-to-head draw rate |
| `h2h_away_wr` | Head-to-head away win rate |
| *(plus team names for lookup)* |

## 📈 Model Performance

Models are evaluated on a held-out test set with metrics including:
- **Accuracy**: Overall correct predictions
- **Precision/Recall/F1**: Per-class performance
- **Classification Report**: Detailed per-target metrics

*The models achieve approximately 55-65% accuracy on test data, which is expected for sports prediction given the inherent unpredictability of football matches.*

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Streamlit Cloud

1. Push this repository to GitHub
2. Create a new app on [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file to `app.py`
5. Deploy!

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

## 📦 Requirements

```
streamlit>=1.30.0
pandas>=2.2.0
numpy>=1.24.0
scikit-learn>=1.4.0
joblib>=1.4.0
```

## 👤 Author
1 . Ahmed Ezzt Thabet


## 🙏 Acknowledgments

- Premier League match data
- Streamlit community for UI components
- SHAP for model explainability