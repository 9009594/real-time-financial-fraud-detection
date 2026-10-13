
import os
import sys
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streaming.fraud_detection import predict_transaction

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "creditcard_features.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "fraud_stream_test_results.csv"
)

df = pd.read_csv(DATA_PATH)

# Select 10 transactions known to be fraudulent
fraud_rows = df[df["Class"] == 1].head(10)

if fraud_rows.empty:
    raise ValueError("No fraud transactions found in dataset!")

results = []

print("\nFraud-Focused Streaming Test")
print("=" * 75)

for count, (row_index, row) in enumerate(fraud_rows.iterrows(), start=1):
    period = row["Time_Period"]

    transaction = row.drop(
        labels=["Class", "Time_Period"]
    ).to_dict()

    transaction["Time_Period_Evening"] = int(period == "Evening")
    transaction["Time_Period_Morning"] = int(period == "Morning")
    transaction["Time_Period_Night"] = int(period == "Night")

    result = predict_transaction(transaction)
    actual_class = int(row["Class"])

    record = {
        "transaction_id": int(row_index) + 1,
        "actual_class": actual_class,
        "predicted_result": result["prediction"],
        "fraud_probability": result["fraud_probability"],
        "risk_alert": result["risk_alert"],
        "correct": result["prediction"] == "FRAUD",
    }

    results.append(record)

    print(
        f"Transaction {count}/10 | "
        f"Prediction: {result['prediction']} | "
        f"Probability: {result['fraud_probability']:.2f} | "
        f"Alert: {result['risk_alert']}"
    )

    time.sleep(1)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

detected = (results_df["predicted_result"] == "FRAUD").sum()
missed = len(results_df) - detected

print("\n" + "=" * 75)
print("Fraud transactions tested:", len(results_df))
print("Frauds detected:", detected)
print("Frauds missed:", missed)
print("Detection rate:", round(detected / len(results_df) * 100, 2), "%")
print("\nAlert counts:")
print(results_df["risk_alert"].value_counts())
print("\nDetailed results:")
print(results_df.to_string(index=False))
print("\nResults saved to:", OUTPUT_PATH)

