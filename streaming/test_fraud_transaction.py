
import os
import sys
import pandas as pd

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streaming.fraud_detection import predict_transaction

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "creditcard_features.csv"
)

# Load dataset and select a known fraud transaction
df = pd.read_csv(DATA_PATH)
fraud_rows = df[df["Class"] == 1]

if fraud_rows.empty:
    raise ValueError("No fraud transactions found in dataset!")

row = fraud_rows.iloc[0]

actual_class = int(row["Class"])
transaction = row.drop(labels=["Class"]).to_dict()

# Apply the same one-hot encoding used during model training
period = transaction.pop("Time_Period")

transaction["Time_Period_Evening"] = int(period == "Evening")
transaction["Time_Period_Morning"] = int(period == "Morning")
transaction["Time_Period_Night"] = int(period == "Night")

# Predict the transaction
result = predict_transaction(transaction)

print("\nKnown Fraud Transaction Test")
print("=" * 40)
print("Actual class:", actual_class)
print("Predicted result:", result["prediction"])
print("Fraud probability:", result["fraud_probability"])
print("Correct prediction:", result["prediction"] == "FRAUD")
