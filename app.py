import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Default AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
MODEL_PATHS = {
    'xgb_json': 'xgboost_model.json',
    'xgb_ubj': 'xgboost_model.ubj',
    'xgb_pkl': 'credit_default_model.pkl'
}
SCALER_PATHS = {
    'new': 'scaler_new.pkl',
    'old': 'scaler.pkl'
}
ASSETS_DIR = "assets"
CSS_FILE = os.path.join(ASSETS_DIR, "style.css")

FEATURE_COLUMNS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

RESULT_CARD_HTML = """
<div class="result-container {css_class}">
    <h2 class="result-title">RISK ASSESSMENT RESULT</h2>
    <div class="result-content">
        <div class="emoji-icon">{emoji}</div>
        <span class="metric-label">PROBABILITY OF DEFAULT</span>
        <div class="risk-score">{probability}</div>
        <div class="risk-level-badge">Risk Level: <strong>{risk_level}</strong></div>
    </div>
    <div class="recommendation">
        <strong class="verdict">{verdict_text}</strong>
    </div>
</div>
"""

# --- Helper Functions ---
def load_file_content(filepath):
    """Helper to read file content."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

@st.cache_resource
def load_model_and_scaler():
    """Load the best available XGBoost model and scaler."""
    model = None
    # Load XGBoost model
    if os.path.exists(MODEL_PATHS['xgb_json']):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATHS['xgb_json'])
    elif os.path.exists(MODEL_PATHS['xgb_ubj']):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATHS['xgb_ubj'])
    elif os.path.exists(MODEL_PATHS['xgb_pkl']):
        model = joblib.load(MODEL_PATHS['xgb_pkl'])

    # Load scaler
    scaler_path = SCALER_PATHS['new'] if os.path.exists(SCALER_PATHS['new']) else SCALER_PATHS['old']
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return model, scaler

def get_risk_assessment(probability):
    """Determine risk level and details based on prediction probability."""
    if probability > 0.7:
        return "status-critical", "⛔ CRITICAL RISK", "Critical", "🔴"
    elif probability > 0.5:
        return "status-high", "⚠️ HIGH RISK", "High", "🟠"
    elif probability > 0.3:
        return "status-medium", "⚡ MODERATE RISK", "Moderate", "🟡"
    else:
        return "status-low", "✅ LOW RISK", "Low", "🟢"

# --- Main Application ---

# Load CSS and models
css_code = load_file_content(CSS_FILE)
st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)
model, scaler = load_model_and_scaler()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='font-size: 24px; text-align: center;'>About</h1>", unsafe_allow_html=True)
    st.markdown("""
        This application uses an **XGBoost** machine learning model to predict the probability of a client defaulting on their credit card payment.
        
        The model was trained on the [UCI Credit Card Default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).
        
        **Built by:** Gemini AI
        **Version:** 1.2.0
    """)
    st.markdown("---")
    st.info("Run `python fix_models.py` if you encounter model version errors.")


# --- Header ---
st.markdown("<div class='main-header'>💳 CREDIT DEFAULT AI</div>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>🚀 Advanced Financial Risk Assessment Engine</p>", unsafe_allow_html=True)
st.markdown("<hr class='header-divider'>", unsafe_allow_html=True)


if not model or not scaler:
    st.error("❌ **Critical Error:** Model or scaler files not found. Please ensure the model files (`xgboost_model.json`, `scaler_new.pkl`, etc.) are in the correct directory. You may need to run `python fix_models.py` first.")
else:
    # --- Input Form ---
    with st.form("credit_input_form"):
        st.markdown("## 📋 Client Information")
        
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            inputs = {}
            inputs['LIMIT_BAL'] = st.number_input("💰 Credit Limit (NT$)", 0, 1000000, 50000)
            inputs['SEX'] = st.selectbox("👥 Gender", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        with c2:
            inputs['EDUCATION'] = st.selectbox("🎓 Education", [1, 2, 3, 4, 0], format_func=lambda x: {0:"Other/Unknown", 1:"Graduate School", 2:"University", 3:"High School", 4:"Other/Unknown"}[x])
            inputs['MARRIAGE'] = st.selectbox("💍 Marital Status", [1, 2, 3, 0], format_func=lambda x: {0:"Other/Unknown", 1:"Married", 2:"Single", 3:"Other/Unknown"}[x])
        with c3:
            inputs['AGE'] = st.number_input("👤 Age", 18, 90, 30)

        st.markdown("---")
        st.markdown("## 📈 Payment History (Last 6 Months)")
        st.markdown("<p style='color: #666; margin-top: -10px;'>Repayment status: -1=Paid Duly, 0=Revolving, 1-8=Payment Delay (months)</p>", unsafe_allow_html=True)
        
        pay_cols = st.columns(6, gap="small")
        pay_months = ["Sep (PAY_0)", "Aug (PAY_2)", "Jul (PAY_3)", "Jun (PAY_4)", "May (PAY_5)", "Apr (PAY_6)"]
        pay_keys = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        for i, col in enumerate(pay_cols):
            with col:
                st.markdown(f"<p class='slider-label'>{pay_months[i].split()[0]}</p>", unsafe_allow_html=True)
                inputs[pay_keys[i]] = st.slider(pay_months[i], -1, 8, 0, key=pay_keys[i], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("## 💵 Financial Details (Last 6 Months in NT$)")
        
        bill_cols, pay_amt_cols = st.columns(2, gap="large")
        with bill_cols:
            st.markdown("#### 📊 Bill Statements")
            bill_keys = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            for i, key in enumerate(bill_keys):
                inputs[key] = st.number_input(f"{pay_months[i].split()[0]} Bill", 0, 1000000, value=0, key=key)

        with pay_amt_cols:
            st.markdown("#### ✅ Previous Payments")
            pay_amt_keys = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            for i, key in enumerate(pay_amt_keys):
                inputs[key] = st.number_input(f"{pay_months[i].split()[0]} Payment", 0, 1000000, value=0, key=key)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Submit Button ---
        submitted = st.form_submit_button("🔍 ANALYZE CREDIT RISK", use_container_width=True)

    # --- Prediction Logic ---
    if submitted:
        # Prepare feature vector in the correct order
        feature_vector = [inputs[col] for col in FEATURE_COLUMNS]
        input_data = pd.DataFrame([feature_vector], columns=FEATURE_COLUMNS)
        
        # Scale data and predict
        scaled_data = scaler.transform(input_data)
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Get assessment and display result
        css_class, verdict, risk_level, emoji = get_risk_assessment(probability)
        
        final_html = RESULT_CARD_HTML.format(
            css_class=css_class,
            probability=f"{probability:.1%}",
            verdict_text=verdict,
            risk_level=risk_level,
            emoji=emoji
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## 📊 Risk Analysis Result")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(final_html, unsafe_allow_html=True)