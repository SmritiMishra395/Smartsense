"""
Train an Isolation Forest model for anomaly detection.

Run after generate_data.py:
    python monitor/ml/train_model.py

Saves both the model and the scaler — you need BOTH at inference time
because Isolation Forest is sensitive to feature scaling.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'sensor_data.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.pkl')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'scaler.pkl')


def train():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run generate_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    features = ['temperature', 'vibration', 'power']
    X = df[features].values
    y_true = df['is_anomaly'].values

    # Scale features (Isolation Forest is sensitive to scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # matches the 5% anomaly rate in our data
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Evaluate on training data
    y_pred_raw = model.predict(X_scaled)
    y_pred = (y_pred_raw == -1).astype(int)  # -1 means anomaly in sklearn

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(f"            Predicted Normal | Predicted Anomaly")
    print(f"Normal     {cm[0][0]:>15} | {cm[0][1]:>16}")
    print(f"Anomaly    {cm[1][0]:>15} | {cm[1][1]:>16}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    # Save both
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}")
    print(f"✓ Scaler saved → {SCALER_PATH}")


if __name__ == '__main__':
    train()
