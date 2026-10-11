import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fraud_predictions.csv")


def save_prediction(transaction_id, actual_class, result):
    """Save one transaction prediction to a CSV log."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    record = pd.DataFrame([{
        "transaction_id": transaction_id,
        "actual_class": actual_class,
        "predicted_result": result["prediction"],
        "fraud_probability": result["fraud_probability"],
    }])

    record.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=not os.path.exists(OUTPUT_PATH),
        index=False
    )

    print(f"Prediction saved: {OUTPUT_PATH}")
