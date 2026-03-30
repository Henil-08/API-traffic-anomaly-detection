import numpy as np
import pandas as pd
import os

def generate_api_traffic(days=14, anomaly_fraction=0.02):
    os.makedirs('data', exist_ok=True)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=days * 24 * 60, freq='min')
    n_samples = len(timestamps)

    # Base patterns: daily seasonality + noise
    time_rad = np.linspace(0, days * 2 * np.pi, n_samples)
    base_req_count = 500 + 300 * np.sin(time_rad) + np.random.normal(0, 50, n_samples)
    base_latency = 120 + 20 * np.cos(time_rad) + np.random.normal(0, 10, n_samples)
    base_error_count = np.abs(np.random.normal(0, 2, n_samples))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'request_count': np.maximum(0, base_req_count),
        'avg_latency': np.maximum(50, base_latency),
        'error_count': np.round(np.maximum(0, base_error_count)),
        'is_anomaly': 0
    })

    # Inject realistic anomalies (traffic spikes, latency hangs, error storms)
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        df.loc[idx, 'is_anomaly'] = 1
        anomaly_type = np.random.choice(['latency_spike', 'traffic_spike', 'error_storm'])
        if anomaly_type == 'latency_spike':
            df.loc[idx, 'avg_latency'] *= np.random.uniform(3, 8)
        elif anomaly_type == 'traffic_spike':
            df.loc[idx, 'request_count'] *= np.random.uniform(2, 5)
        else:
            df.loc[idx, 'error_count'] += np.random.randint(50, 200)

    df.to_csv('data/api_traffic.csv', index=False)
    print(f"Generated {n_samples} records. Saved to data/api_traffic.csv")

if __name__ == "__main__":
    generate_api_traffic()