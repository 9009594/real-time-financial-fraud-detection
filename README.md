# Real-Time Financial Fraud Detection Pipeline

A machine learning project that analyzes financial transactions, predicts potential fraud, and generates risk alerts using Python and a Random Forest classifier.

## Project Overview

This project demonstrates an end-to-end fraud detection workflow, including data cleaning, feature engineering, model training, transaction prediction, threshold-based alerting, and prediction logging.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy, Jupyter Notebook
- **Machine Learning:** Scikit-learn, Random Forest
- **Data Visualization:** Matplotlib, Seaborn
- **Version Control:** Git and GitHub

## Project Structure

```text
data/
  raw/
  processed/
producer/
streaming/
models/
database/
notebooks/
monitoring/
tests/
docs/
screenshots/
requirements.txt
README.md
```

data/ — Raw and processed transaction data
producer/ — Transaction producer
streaming/ — Prediction, streaming tests, and alert logic
models/ — Trained model and feature configuration
notebooks/ — Data understanding, cleaning, feature engineering, and model training
database/ — Database-related components
monitoring/ — Monitoring components
tests/ — Project tests
docs/ — Project documentation
screenshots/ — Project screenshots

## Model Performance

The Random Forest model was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Precision | 97.18% |
| Recall | 72.63% |
| F1-score | 83.13% |
| Accuracy | 99.95% |

These metrics use the model's original 0.50 classification threshold. Accuracy alone can be misleading for imbalanced fraud datasets, so precision and recall are also reported.

## Fraud Detection and Alert Logic

The current prediction pipeline uses a 0.40 classification threshold.

- **NORMAL:** Fraud probability below 0.40
- **FRAUD_REVIEW:** Fraud probability from 0.40 to below 0.70
- **HIGH_RISK_ALERT:** Fraud probability of 0.70 or above

Predictions and risk alerts are logged to a processed CSV file for review.

## Run the Project

1. Clone the repository and open the project folder in VS Code.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Run the prediction module's tests:

   ```powershell
   python streaming/test_threshold_alerts.py
   ```

5. Run the transaction producer:

   ```powershell
   python producer/transaction_producer.py
   ```

Run commands from the project root. If a script uses local module imports, run it using the invocation supported by that script.

## Dataset

The project uses a credit card transaction dataset with anonymized features, transaction amount, time, and a fraud label. The dataset is highly imbalanced, reflecting the challenge of identifying rare fraudulent transactions.

## Disclaimer

This is an educational portfolio project. Its predictions and thresholds are not validated for real-world financial decisions or production deployment.

## Current Status

Core data preparation, feature engineering, model training, prediction, threshold-based alerting, CSV logging, and threshold tests have been implemented. Further work can include database integration, live message-broker streaming, monitoring dashboards, and production deployment.