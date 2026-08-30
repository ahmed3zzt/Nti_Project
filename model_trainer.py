import os
import streamlit as st
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_resource
def load_models():
    result_model = joblib.load(os.path.join(MODEL_DIR, "best_match_predictor_model.pkl"))
    result_le = joblib.load(os.path.join(MODEL_DIR, "label_encoder_result.pkl"))
    result_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_result.pkl"))
    result_features = joblib.load(os.path.join(MODEL_DIR, "feature_names_result.pkl"))

    ou_model = joblib.load(os.path.join(MODEL_DIR, "over_under_model.pkl"))
    ou_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_ou.pkl"))
    ou_features = joblib.load(os.path.join(MODEL_DIR, "feature_names_ou.pkl"))

    btts_model = joblib.load(os.path.join(MODEL_DIR, "btts_model.pkl"))
    btts_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_btts.pkl"))
    btts_features = joblib.load(os.path.join(MODEL_DIR, "feature_names_btts.pkl"))

    return {
        "result": (result_model, result_le, result_scaler, result_features),
        "ou": (ou_model, ou_scaler, ou_features),
        "btts": (btts_model, btts_scaler, btts_features),
    }
