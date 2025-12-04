import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Default AI", page_icon="💳", layout="wide", initial_sidebar_state="collapsed")

def load_file(filepath):
    """Helper to read CSS/HTML files"""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

css_code = load_file(os.path.join("assets", "style.css"))
st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('credit_default_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_models()

# Header with improved styling
header_col1, header_col2 = st.columns([0.5, 4])
with header_col1:
    st.markdown("<div style='font-size: 48px; text-align: center;'>💳</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown("<h1 style='margin: 0; padding: 0;'>CREDIT DEFAULT AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; margin: 5px 0 0 0; font-size: 14px;'>🚀 Advanced Financial Risk Assessment Engine | XGBoost + StandardScaler</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, #000, transparent); margin: 20px 0;'></div>", unsafe_allow_html=True)

if model:
    # Client Data Section
    st.markdown("<h2 style='margin-top: 0;'>📋 CLIENT INFORMATION</h2>", unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4, gap="large")
        with c1:
            st.markdown("<span style='font-weight: bold;'>💰 Credit Limit</span>", unsafe_allow_html=True)
            limit_bal = st.number_input("LIMIT_BAL", 0, 1000000, 50000, label_visibility="collapsed")
            st.markdown("<span style='font-weight: bold; color: #333;'>👤 Age</span>", unsafe_allow_html=True)
            age = st.number_input("AGE", 18, 90, 30, label_visibility="collapsed")
        with c2:
            st.markdown("<span style='font-weight: bold; color: #333;'>👥 Gender</span>", unsafe_allow_html=True)
            sex = st.selectbox("SEX", [1, 2], format_func=lambda x: "Male" if x==1 else "Female", label_visibility="collapsed")
            st.markdown("<span style='font-weight: bold; color: #333;'>🎓 Education</span>", unsafe_allow_html=True)
            edu = st.selectbox("EDUCATION", [1, 2, 3, 4, 5, 6, 0], format_func=lambda x: {0:"Unknown", 1:"Grad School", 2:"University", 3:"High School", 4:"Others", 5:"Others", 6:"Others"}[x], label_visibility="collapsed")
        with c3:
            st.markdown("<span style='font-weight: bold; color: #333;'>💍 Marital Status</span>", unsafe_allow_html=True)
            marriage = st.selectbox("MARRIAGE", [0, 1, 2, 3], format_func=lambda x: {0:"Unknown", 1:"Married", 2:"Single", 3:"Others"}[x], label_visibility="collapsed")
        with c4:
            st.markdown("<span style='color: transparent;'>_</span>", unsafe_allow_html=True)  # Spacer
            st.markdown("<span style='color: transparent;'>_</span>", unsafe_allow_html=True)  # Spacer

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Payment History Section
    st.markdown("<h2>📈 PAYMENT HISTORY (Last 6 Months)</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<p style='color: #666; margin: -10px 0 15px 0;'>Payment Status: -1=Pay duly, 1=Payment delay for one month, 2=Payment delay for two months, etc.</p>", unsafe_allow_html=True)
        cols = st.columns(6, gap="small")
        pay_status = []
        months = ["Sep (PAY_0)", "Aug (PAY_2)", "Jul (PAY_3)", "Jun (PAY_4)", "May (PAY_5)", "Apr (PAY_6)"]
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"<p style='text-align: center; font-weight: bold; margin: 0 0 5px 0;'>{months[i].split()[0]}</p>", unsafe_allow_html=True)
                val = st.slider(f"{months[i]}", -1, 8, 0, key=f"pay_{i}", label_visibility="collapsed")
                pay_status.append(val)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Financial Details Section
    st.markdown("<h2>💵 FINANCIAL DETAILS</h2>", unsafe_allow_html=True)
    with st.container():
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("<h4 style='margin-bottom: 15px;'>📊 Bill Statements (NT$)</h4>", unsafe_allow_html=True)
            bills = []
            for i, m in enumerate(months):
                label = m.split("(")[1].rstrip(")")
                bills.append(st.number_input(f"{label}", 0, 1000000, value=0, key=f"bill_{i}"))
        with col_b:
            st.markdown("<h4 style='margin-bottom: 15px;'>✅ Payment Amounts (NT$)</h4>", unsafe_allow_html=True)
            pays = []
            for i, m in enumerate(months):
                label = m.split("(")[1].rstrip(")").replace("PAY", "PAY_AMT")
                pays.append(st.number_input(f"{label}", 0, 1000000, value=0, key=f"paid_{i}"))

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Prediction Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1.5, 1])
    with btn_col2:
        run_prediction = st.button("🔍 ANALYZE CREDIT RISK", use_container_width=True)

    if run_prediction:
        # Prepare Feature Vector in correct order (matching the notebook feature order)
        features = [limit_bal, sex, edu, marriage, age] + pay_status + bills + pays
        
        # Create DataFrame with exact column names from UCI Credit Card dataset
        input_data = pd.DataFrame([features], columns=[
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ])
        
        # Apply StandardScaler (same as in notebook)
        scaled_data = scaler.transform(input_data)
        
        # Get prediction from XGBoost model
        prob = model.predict_proba(scaled_data)[0][1]
        
        html_template = load_file(os.path.join("assets", "result_card.html"))
        
        # Enhanced risk assessment with 4 levels
        if prob > 0.7:
            css_class = "status-critical"
            verdict = "⛔ CRITICAL RISK - DO NOT APPROVE"
            risk_level = "Critical"
            emoji = "🔴"
        elif prob > 0.5:
            css_class = "status-high"
            verdict = "⚠️ HIGH RISK - REQUIRES REVIEW"
            risk_level = "High"
            emoji = "🟠"
        elif prob > 0.3:
            css_class = "status-medium"
            verdict = "⚡ MODERATE RISK - CAUTION ADVISED"
            risk_level = "Moderate"
            emoji = "🟡"
        else:
            css_class = "status-low"
            verdict = "✅ LOW RISK - APPROVAL RECOMMENDED"
            risk_level = "Low"
            emoji = "🟢"

        final_html = html_template.format(
            css_class=css_class,
            probability=f"{prob:.1%}",
            verdict_text=verdict,
            risk_level=risk_level,
            emoji=emoji
        )
        
        # Render Result with enhanced styling
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(final_html, unsafe_allow_html=True)

else:
    st.error("System Error: Model files (pkl) not found in root directory.")