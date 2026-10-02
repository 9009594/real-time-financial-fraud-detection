import pandas as pd
import time
import json

DATA_PATH = "data/processed/creditcard_feature_engineered.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total transactions:", len(df))

sample_transactions = df.sample(n=20, random_state=42).reset_index(drop=True)


print("\nStarting real-time transaction stream...")
print("=" * 60)

for index, row in sample_transactions.iterrows():
    transaction = {
        "transaction_id": index + 1,
        "Time": float(row["Time"]),
        "Amount": float(row["Amount"]),
        "V1": float(row["V1"]),
        "V2": float(row["V2"]),
        "V3": float(row["V3"]),
        "V4": float(row["V4"]),
        "V5": float(row["V5"]),
        "V6": float(row["V6"]),
        "V7": float(row["V7"]),
        "V8": float(row["V8"]),
        "V9": float(row["V9"]),
        "V10": float(row["V10"]),
        "V11": float(row["V11"]),
        "V12": float(row["V12"]),
        "V13": float(row["V13"]),
        "V14": float(row["V14"]),
        "V15": float(row["V15"]),
        "V16": float(row["V16"]),
        "V17": float(row["V17"]),
        "V18": float(row["V18"]),
        "V19": float(row["V19"]),
        "V20": float(row["V20"]),
        "V21": float(row["V21"]),
        "V22": float(row["V22"]),
        "V23": float(row["V23"]),
        "V24": float(row["V24"]),
        "V25": float(row["V25"]),
        "V26": float(row["V26"]),
        "V27": float(row["V27"]),
        "V28": float(row["V28"]),
        "Log_Amount": float(row["Log_Amount"]),
        "Class": int(row["Class"]),
    }

    print(json.dumps(transaction, indent=2))

    print("-" * 60)

    # Simulate real-time delay
    time.sleep(1)

print("\nTransaction stream completed!")
