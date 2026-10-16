import os
import sys
import math
import sqlite3
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streaming.fraud_detection import predict_transaction
from database.database_manager import (
    initialize_database,
    save_prediction_to_database,
    DATABASE_PATH,
)

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Real-Time Financial Fraud Detection")
st.write("Random Forest based transaction risk analysis dashboard.")

initialize_database()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
        transaction_time = st.number_input(
            "Time (seconds)", min_value=0.0, value=10000.0
        )
    with col2:
        hour = st.slider("Transaction Hour", 0, 23, 12)
        period = st.selectbox(
            "Time Period", ["Morning", "Afternoon", "Evening", "Night"]
        )

    submitted = st.form_submit_button("Predict Fraud Risk", type="primary")

if submitted:
    transaction = {
        "Time": float(transaction_time),
        "Amount": float(amount),
        "Hour": int(hour),
        "Log_Amount": math.log1p(amount),
        "Time_Period_Evening": int(period == "Evening"),
        "Time_Period_Morning": int(period == "Morning"),
        "Time_Period_Night": int(period == "Night"),
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    result = predict_transaction(transaction)

    with sqlite3.connect(DATABASE_PATH) as conn:
        transaction_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM transaction_predictions"
        ).fetchone()[0]

    save_prediction_to_database(transaction_id, -1, result)

    a, b = st.columns(2)
    a.metric("Prediction", result["prediction"])
    b.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")

    if result["risk_alert"] == "HIGH_RISK_ALERT":
        st.error("🚨 HIGH RISK ALERT")
    elif result["risk_alert"] == "FRAUD_REVIEW":
        st.warning("⚠️ FRAUD REVIEW")
    else:
        st.success("✅ NORMAL")

    st.caption(
        "Demo limitation: V1–V28 are set to zero. This input is not a validated "
        "real-world banking transaction, so predictions are illustrative only."
    )

st.divider()
st.subheader("Prediction History")

with sqlite3.connect(DATABASE_PATH) as conn:
    history = pd.read_sql_query(
        """
        SELECT transaction_id, predicted_result, fraud_probability,
               risk_alert, created_at
        FROM transaction_predictions
        ORDER BY id DESC
        LIMIT 100
        """,
        conn,
    )

if history.empty:
    st.info("No predictions saved yet.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Recent Records", len(history))
    c2.metric("Fraud Predictions", int((history["predicted_result"] == "FRAUD").sum()))
    c3.metric(
        "High-Risk Alerts", int((history["risk_alert"] == "HIGH_RISK_ALERT").sum())
    )
    st.dataframe(history, use_container_width=True, hide_index=True)
