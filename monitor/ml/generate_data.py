"""
Generate simulated industrial sensor data.

Run this once before training:
    python monitor/ml/generate_data.py

It creates sensor_data.csv with 1000 readings, 5% of which are anomalies
(temperature spikes, vibration spikes, or power drops).
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Allow running standalone
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'sensor_data.csv')


def generate_sensor_data(n_samples=1000, anomaly_fraction=0.05, seed=42):
    """
    Simulate 3 industrial sensors over time.

    Normal operating ranges:
      temperature: 18–28°C  (mean=23, std=2)
      vibration:    0–5 mm/s (mean=2.5, std=0.8)
      power:      100–500 W  (mean=300, std=50)

    Returns a DataFrame with timestamp, sensor readings, and is_anomaly label.
    """
    np.random.seed(seed)
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
    timestamps.reverse()

    # Normal data
    temperature = np.random.normal(23, 2, n_samples)
    vibration = np.random.normal(2.5, 0.8, n_samples)
    power = np.random.normal(300, 50, n_samples)

    labels = np.zeros(n_samples, dtype=int)

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for i in anomaly_indices:
        anomaly_type = np.random.choice(['temp_spike', 'vibration_spike', 'power_drop'])
        if anomaly_type == 'temp_spike':
            temperature[i] = np.random.uniform(38, 50)
        elif anomaly_type == 'vibration_spike':
            vibration[i] = np.random.uniform(12, 20)
        else:  # power_drop
            power[i] = np.random.uniform(10, 40)
        labels[i] = 1

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.clip(temperature, 0, 60).round(2),
        'vibration': np.clip(vibration, 0, 25).round(2),
        'power': np.clip(power, 0, 700).round(2),
        'is_anomaly': labels
    })

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Generated {n_samples} samples ({n_anomalies} anomalies) → {OUTPUT_PATH}")
    return df


if __name__ == '__main__':
    generate_sensor_data()
