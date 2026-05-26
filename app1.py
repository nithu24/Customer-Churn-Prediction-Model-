import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    layout="centered"
)

# ------------------ BACKGROUND IMAGE ------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🔴 Place your image in the same folder
add_bg_from_local("image_1.png")


# ------------------ LOAD MODEL ARTIFACTS ------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#b30000;'>📊 Telecom Customer Churn Predictor</h1>
    <p style='text-align:center; color:#8b0000; font-size:20px;'>
    AI-powered application to identify customers at risk of churn
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("🧑‍💼 Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)

online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# ------------------ FEATURE ENGINEERING ------------------
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "OnlineSecurity": 1 if online_security == "Yes" else 0,
    "OnlineBackup": 1 if online_backup == "Yes" else 0,
    "DeviceProtection": 1 if device_protection == "Yes" else 0,
    "StreamingTV": 1 if streaming_tv == "Yes" else 0,
    "PaperlessBilling": 1 if paperless == "Yes" else 0,

    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,

    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,

    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
}

input_df = pd.DataFrame([input_dict])

# Align with training features
for col in feature_columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[feature_columns]

# ------------------ MAIN CARD ------------------
# st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📈 Prediction Result")

if st.button("🔍 Predict Churn"):
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    # st.write(f"### Churn Probability: **{prob:.2%}**")
    # Decide color & message
    if prob > 0.7:
        color = "#b30000"  # dark red
        message = "🚨 High risk of churn. Immediate retention action recommended."
    elif prob > 0.4:
        color = "#ff8c00"  # orange
        message = "⚠️ Moderate churn risk. Monitor customer closely."
    else:
        color = "#006400"  # dark green
        message = "✅ Low churn risk. Customer likely to stay."

    # Styled prediction card
    st.markdown(
        f"""
        <div style="
            padding:12px;
            border-radius:10px;
            border:1px solid {color};
            background-color:#ffe4e1;
            text-align:center;
            margin-top:10px;
        ">
            <h2 style="color:{color};">
                Churn Probability: {prob:.2%}
            </h2>
            <h4 style="color:{color};">
                {message}
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(int(prob * 100))
    st.markdown(
        """
        <style>
        /* Progress bar container */
        .stProgress > div > div {
            background-color: #ffd1dc !important;  /* light pink track */
        }

        /* Progress bar fill */
        .stProgress > div > div > div {
            background-color: #ff4d6d !important;  /* pink fill */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # if prob > 0.7:
    #     st.error("🚨 High risk of churn. Immediate retention action recommended.")
    # elif prob > 0.4:
    #     st.warning("⚠️ Moderate churn risk. Monitor customer closely.")
    # else:
    #     st.success("✅ Low churn risk. Customer likely to stay.")

    with st.expander("📋 View Model Input Features"):
        st.dataframe(input_df)
st.markdown(
    """
    <style>
    /* Expander container */
    div[data-testid="stExpander"] {
        background-color: #ffe4e1;   /* light pink */
        border-radius: 12px;
        border: 1px solid #ff69b4;
    }

    /* Expander header text */
    div[data-testid="stExpander"] > div:first-child {
        color: #b30000;
        font-weight: bold;
        font-size: 16px;
    }

    /* Expander body text */
    div[data-testid="stExpander"] div[role="region"] {
        color: #4a0000;
    }

    /* Dataframe text */
    .stDataFrame {
        color: #4a0000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown(
    """
    <p style='text-align:center; color:black; font-size:16px;'>
    © 2025 | Telecom Churn Prediction | Powered by Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)
