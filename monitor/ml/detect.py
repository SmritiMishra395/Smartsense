"""
Real-time anomaly detection inference.

Loads the trained model once and exposes detect_anomaly() which returns
a structured result the Django view + AI agent both consume.
"""
import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.pkl')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'scaler.pkl')

# Lazy-load model/scaler so Django startup doesn't fail if they're not yet built
_model = None
_scaler = None


def _ensure_loaded():
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python monitor/ml/generate_data.py && python monitor/ml/train_model.py"
            )
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        logger.info("ML model loaded successfully")


def detect_anomaly(temperature: float, vibration: float, power: float) -> dict:
    """
    Run anomaly detection on a single sensor reading.

    Returns a dict with:
      is_anomaly: bool
      score: float (more negative = more anomalous)
      anomaly_features: list[str] — human-readable description of out-of-range sensors
      readings: dict — the original readings
    """
    _ensure_loaded()

    X = np.array([[temperature, vibration, power]])
    X_scaled = _scaler.transform(X)

    prediction = _model.predict(X_scaled)[0]      # -1 or 1
    score = float(_model.score_samples(X_scaled)[0])

    is_anomaly = prediction == -1

    # Identify which specific sensors look out of range (used by agent for context)
    anomaly_features = []
    if temperature > 35 or temperature < 10:
        anomaly_features.append(f"temperature={temperature:.1f}°C (normal: 18-28°C)")
    if vibration > 8:
        anomaly_features.append(f"vibration={vibration:.1f}mm/s (normal: 0-5mm/s)")
    if power < 60 or power > 600:
        anomaly_features.append(f"power={power:.1f}W (normal: 100-500W)")

    return {
        'is_anomaly': bool(is_anomaly),
        'score': score,
        'anomaly_features': anomaly_features,
        'readings': {
            'temperature': round(temperature, 2),
            'vibration': round(vibration, 2),
            'power': round(power, 2)
        }
    }
