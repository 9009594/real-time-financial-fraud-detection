
import os
import sys
import time
import json
import pandas as pd

# Allow importing the fraud detection module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streaming.fraud_detection import predict_transaction

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "creditcard_features.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total transactions:", len(df))

sample_transactions = df.sample(n=20, random_state=42).reset_index(drop=True)

print("\nStarting real-time fraud detection stream...")
print("=" * 60)

for index, row in sample_transactions.iterrows():
    period = row["Time_Period"]

    transaction = {
        "Time": float(row["Time"]),
        "Amount": float(row["Amount"]),
        "Hour": int(row["Hour"]),
        "Log_Amount": float(row["Log_Amount"]),
        "Time_Period_Evening": int(period == "Evening"),
        "Time_Period_Morning": int(period == "Morning"),
        "Time_Period_Night": int(period == "Night"),
    }

    # Add anonymized transaction features V1 to V28
    for feature_num in range(1, 29):
        column = f"V{feature_num}"
        transaction[column] = float(row[column])

    # Run the fraud detection model
    result = predict_transaction(transaction)

    print(json.dumps({
        "transaction_id": index + 1,
        "actual_class": int(row["Class"]),
        "predicted_result": result["prediction"],
        "fraud_probability": result["fraud_probability"],
    }, indent=2))

    print("-" * 60)

    # Simulate a one-second real-time delay
    time.sleep(1)

print("\nTransaction stream completed!")
