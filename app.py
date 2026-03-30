import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="API Anomaly Detection", layout="wide")
st.title("Real-Time API Traffic & Anomaly Detection")

@st.cache_data
def load_data():
    try:
        return pd.read_csv('../data/predictions.csv')
    except FileNotFoundError:
        st.error("Please run src/train.py first to generate predictions.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.header("Threshold Settings")
    # Dynamic threshold based on percentiles of reconstruction error
    threshold_pct = st.sidebar.slider("Anomaly Threshold (Percentile)", 80.0, 99.9, 95.0)
    threshold_val = np.percentile(df['reconstruction_error'], threshold_pct)
    
    df['predicted_anomaly'] = df['reconstruction_error'] > threshold_val
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total API Requests Assessed", f"{len(df):,}")
    col2.metric("Anomalies Detected", int(df['predicted_anomaly'].sum()))
    col3.metric("Error Threshold (MSE)", f"{threshold_val:.4f}")

    # Plot Latency
    st.subheader("API Latency (ms) with Detected Anomalies")
    fig = go.Figure()
    
    # Normal traffic line
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['avg_latency'], 
                             mode='lines', name='Avg Latency', line=dict(color='blue')))
    
    # Anomalies scatter
    anomalies = df[df['predicted_anomaly']]
    fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['avg_latency'],
                             mode='markers', name='Anomaly', 
                             marker=dict(color='red', size=8, symbol='x')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Data View
    st.subheader("Detected Anomaly Logs")
    st.dataframe(anomalies[['timestamp', 'request_count', 'avg_latency', 'error_count', 'reconstruction_error']].sort_values('reconstruction_error', ascending=False))