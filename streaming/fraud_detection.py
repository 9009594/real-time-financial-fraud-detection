
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

print("Fraud detection model loaded successfully!")
print("Feature columns loaded:", len(feature_columns))



def predict_transaction(transaction):
    """Predict transaction status and generate a risk alert."""

    transaction_df = pd.DataFrame([transaction])
    transaction_df = transaction_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    fraud_probability = model.predict_proba(transaction_df)[0][1]
    probability = round(float(fraud_probability), 4)
    prediction = 1 if probability >= 0.40 else 0

    if prediction == 1 and probability >= 0.70:
        risk_alert = "HIGH_RISK_ALERT"
    elif prediction == 1:
        risk_alert = "FRAUD_REVIEW"
    else:
        risk_alert = "NORMAL"

    return {
        "prediction": "FRAUD" if prediction == 1 else "NORMAL",
        "fraud_probability": probability,
        "risk_alert": risk_alert
    }


if __name__ == "__main__":
    DATA_PATH = os.path.join(
        BASE_DIR, "data", "processed", "creditcard_features.csv"
    )

    test_data = pd.read_csv(DATA_PATH)
    test_row = test_data.iloc[0]
    actual_class = int(test_row["Class"])
    transaction = test_row.drop(labels=["Class"]).to_dict()

    period = transaction.pop("Time_Period")
    transaction["Time_Period_Evening"] = int(period == "Evening")
    transaction["Time_Period_Morning"] = int(period == "Morning")
    transaction["Time_Period_Night"] = int(period == "Night")

    result = predict_transaction(transaction)

    print("\nTransaction Prediction Test")
    print("----------------------------")
    print("Actual class:", actual_class)
    print("Predicted result:", result["prediction"])
    print("Fraud probability:", result["fraud_probability"])
