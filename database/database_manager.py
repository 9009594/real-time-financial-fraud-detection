import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "data", "processed")
DATABASE_PATH = os.path.join(DATABASE_DIR, "fraud_detection.db")


def get_connection():
    """Create a connection to the SQLite database."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    """Create the transaction predictions table if it does not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transaction_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                actual_class INTEGER NOT NULL,
                predicted_result TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_alert TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_prediction_to_database(transaction_id, actual_class, result):
    """Save a prediction and risk alert to SQLite."""
    initialize_database()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO transaction_predictions (
                transaction_id,
                actual_class,
                predicted_result,
                fraud_probability,
                risk_alert,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(transaction_id),
                int(actual_class),
                str(result["prediction"]),
                float(result["fraud_probability"]),
                str(result["risk_alert"]),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    print("Prediction saved to SQLite database.")


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DATABASE_PATH}")
