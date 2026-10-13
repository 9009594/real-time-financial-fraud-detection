
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "creditcard_features.csv"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "fraud_detection_model.pkl"
)
FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "feature_columns.pkl"
)

# Load dataset and saved model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# Prepare features exactly as in the pipeline
X = df.drop(columns=["Class"]).copy()
y = df["Class"].astype(int)

X = pd.get_dummies(
    X,
    columns=["Time_Period"],
    drop_first=True,
    dtype=int
)

X = X.reindex(columns=feature_columns, fill_value=0)

# Recreate the held-out split used during model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Calculate fraud probabilities once; only the decision threshold changes
probabilities = model.predict_proba(X_test)[:, 1]

print("\nFraud Detection Threshold Analysis")
print("=" * 75)
print("Test transactions:", len(y_test))
print("Actual fraud transactions:", int(y_test.sum()))
print("Actual normal transactions:", int((y_test == 0).sum()))

for threshold in [0.50, 0.40, 0.30]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"\n--- Threshold: {threshold:.2f} ---")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 score:", round(f1, 4))
    print("True positives (frauds detected):", tp)
    print("False negatives (frauds missed):", fn)
    print("False positives (normal flagged as fraud):", fp)
    print("True negatives (normal correctly classified):", tn)

