import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streaming.fraud_detection import predict_transaction

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "creditcard_features.csv")

df = pd.read_csv(DATA_PATH)

fraud_rows = df[df["Class"] == 1].head(5)
normal_rows = df[df["Class"] == 0].head(5)

test_rows = pd.concat([fraud_rows, normal_rows])

results = []

for index, row in test_rows.iterrows():
    transaction = row.drop(labels=["Class", "Time_Period"]).to_dict()

    period = row["Time_Period"]
    transaction["Time_Period_Evening"] = int(period == "Evening")
    transaction["Time_Period_Morning"] = int(period == "Morning")
    transaction["Time_Period_Night"] = int(period == "Night")

    result = predict_transaction(transaction)
    actual = "FRAUD" if int(row["Class"]) == 1 else "NORMAL"

    results.append(
        {
            "actual": actual,
            "predicted": result["prediction"],
            "fraud_probability": result["fraud_probability"],
            "risk_alert": result["risk_alert"],
            "correct": actual == result["prediction"],
        }
    )

results_df = pd.DataFrame(results)

print("\nMultiple Transaction Test")
print("=" * 65)
print(results_df.to_string(index=False))
print("\nCorrect predictions:", results_df["correct"].sum(), "out of", len(results_df))
print("Test accuracy:", round(results_df["correct"].mean() * 100, 2), "%")
