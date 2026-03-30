# API-traffic-anomaly-detection

This repository implements an end-to-end MLOps pipeline for forecasting API telemetry and detecting system anomalies (latency spikes, traffic drops, error storms) using a PyTorch LSTM-Autoencoder.

## Architecture

1. **Data Pipeline:** Aggregates raw API Gateway logs (simulated via `generate_data.py`) into time-series sequences. 
2. **Modeling:** A PyTorch-based LSTM-Autoencoder compresses traffic windows and flags sequences with high reconstruction errors (MSE) as anomalous.
3. **Tracking:** Training runs, hyperparameters, and ROC-AUC metrics are versioned and logged via MLflow.
4. **Reporting:** A low-latency Streamlit application visualizes traffic trends and democratizes AI insights for stakeholders.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the telemetry data:
```bash
python src/generate_data.py
```

3. Train the model and calculate metrics:
```bash
python src/train.py
```

4. Launch the app:
```bash
cd app
streamlit run dashboard.py
```