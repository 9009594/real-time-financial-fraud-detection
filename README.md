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

Fraud Detection with Deep Learning: An Overview

Fraud is a prevalent problem in virtually all industries, especially in financial services. Businesses lose millions of dollars every year through various forms of fraud such as credit card fraud, insurance fraud, and fake identities. The need for advanced surveillance systems to detect fraud is thus paramount. Fraudulent activities are often complex and hard to detect through traditional methods. Deep Learning (DL) models have emerged as powerful tools for detecting fraud in real-time. In this blog post, we share an overview of DL in fraud detection and how to develop DL-based models to combat fraud.

What is Deep Learning?

Deep Learning is a subfield of Machine Learning (ML), which enables systems to learn and improve from experience. DL encompasses Neural Networks (NN), a type of artificial intelligence technology that functions similarly to the human brain. NN uses layers of interconnected nodes (or neurons) to learn specific data representations. Through a training process, the NN identifies patterns and structures in data which it then uses to learn how to classify, cluster, or predict new data.

How does Deep Learning detect fraud?

Deep Learning is a supervised Machine Learning technique, which means that the DL models require a labeled dataset to learn patterns and structures. When used in fraud detection, DL models use historical data, which has already been labeled as fraud or non-fraud, to learn about patterns and associations between fraudulent and legitimate transactions. The DL model then applies the learned knowledge to new, unlabeled data to determine whether the transaction is fraudulent or not.

What are some examples of Deep Learning-based fraud detection?

There are different types of fraud detection models, each with a unique set of features and design choices that suit distinct applications’ needs. Here are some examples of DL-based models commonly used in fraud detection:

Recurrent Neural Networks (RNN)
RNN is a class of neural networks that have been successfully used in detecting fraudulent activities in sequence data. Sequential data refers to data that has an intrinsic order or a temporal nature that requires a specific analysis approach. In fraud detection, the RNN receives inputs that represent the sequential nature of data, such as transaction timestamps or IP addresses. The RNN then ‘remembers’ past inputs through its hidden layer state, allowing it to identify temporal patterns or discrepancies between legitimate and fraudulent transactions.

Convolutional Neural Networks (CNN)
CNN is a type of neural network that uses convolutional layers to identify spatial patterns in data such as images or graphs. In fraud detection, CNN models can be used to examine transactions, network diagrams, or financial graphs, which exhibit complex spatial patterns associated with fraudulent activities. CNN models are typically used in conjunction with other DL models to leverage the spatial and contextual information within financial data.

Autoencoders (AE)
AE is an unsupervised neural network that is used to encode and decode data through a compressed representation. AE is used in anomaly detection tasks, such as fraud detection, to learn the latent structure of the data and identify outliers that deviate from the norm. In fraud detection, AE models can be used to detect credit card fraud in real-time. AE models reconstruct the input data with the encoder and the decoder layers. In cases where the AE model cannot reconstruct the input data or produces an output that differs significantly from the original data, the transaction is marked as fraudulent.

Conclusion

In conclusion, the increasing complexities in fraud detection processes and the high volumes of data generated from various transactions require advanced and scalable security tools like Deep Learning. This technology can effectively detect fraud in real-time, improve customer experience, and prevent financial losses. We can see that deep learning-based fraud detection systems are being continually designed and refined, and its impact on financial services has been substantial. The ability to detect and prevent fraud is crucial, and DL models continue to be a promising and useful tool in achieving these goals.

## Disclaimer

This is an educational portfolio project. Its predictions and thresholds are not validated for real-world financial decisions or production deployment.

## Current Status

Core data preparation, feature engineering, model training, prediction, threshold-based alerting, CSV logging, and threshold tests have been implemented. Further work can include database integration, live message-broker streaming, monitoring dashboards, and production deployment.
