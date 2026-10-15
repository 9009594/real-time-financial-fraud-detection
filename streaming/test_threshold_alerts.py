import fraud_detection as fd

test_cases = [
    (0.34, "NORMAL", "NORMAL"),
    (0.40, "FRAUD", "FRAUD_REVIEW"),
    (0.63, "FRAUD", "FRAUD_REVIEW"),
    (0.91, "FRAUD", "HIGH_RISK_ALERT"),
]

original_predict_proba = fd.model.predict_proba

try:
    print("\nThreshold-Based Alert Testing")
    print("=" * 65)

    for probability, expected_prediction, expected_alert in test_cases:
        fd.model.predict_proba = lambda X, p=probability: [[1 - p, p]]

        result = fd.predict_transaction({})

        passed = (
            result["prediction"] == expected_prediction
            and result["risk_alert"] == expected_alert
        )

        print(
            f"Probability: {probability:.2f} | "
            f"Prediction: {result['prediction']} | "
            f"Alert: {result['risk_alert']} | "
            f"{'PASS' if passed else 'FAIL'}"
        )

finally:
    fd.model.predict_proba = original_predict_proba

print("=" * 65)
