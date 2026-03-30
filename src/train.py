import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch
import os
from model import LSTMAutoencoder

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

def train():
    mlflow.set_experiment("API_Traffic_Anomaly_Detection")
    
    with mlflow.start_run() as run:
        # 1. Load and Prepare Data
        df = pd.read_csv('data/api_traffic.csv')
        features = ['request_count', 'avg_latency', 'error_count']
        
        # Train on first 70% (assume normal-ish), test on last 30%
        split_idx = int(len(df) * 0.7)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[features])
        test_scaled = scaler.transform(test_df[features])
        
        seq_len = 30
        X_train, _ = create_sequences(train_scaled, train_df['is_anomaly'].values, seq_len)
        X_test, y_test = create_sequences(test_scaled, test_df['is_anomaly'].values, seq_len)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        
        # 2. Setup Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMAutoencoder(seq_len=seq_len, n_features=len(features)).to(device)
        
        epochs = 20
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        mlflow.log_params({"seq_len": seq_len, "epochs": epochs, "learning_rate": lr})
        
        # 3. Train Model
        print("Training model...")
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            X_batch = X_train_t.to(device)
            reconstructed = model(X_batch)
            loss = criterion(reconstructed, X_batch)
            loss.backward()
            optimizer.step()
            
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
                
        # 4. Evaluate and Calculate ROC-AUC
        print("Evaluating on test set...")
        model.eval()
        with torch.no_grad():
            X_test_batch = X_test_t.to(device)
            pred = model(X_test_batch)
            
            # Reconstruction error (MSE per sequence)
            mse = torch.mean(torch.pow(X_test_batch - pred, 2), dim=(1, 2)).cpu().numpy()
            
        # The MSE is our anomaly score. Compare against true labels.
        roc_auc = roc_auc_score(y_test, mse)
        print(f"Real ROC-AUC achieved: {roc_auc:.4f}")
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Save model and artifacts for the dashboard
        mlflow.pytorch.log_model(model, "model")
        
        # Save results for the dashboard to read
        results_df = test_df.iloc[seq_len:].copy()
        results_df['reconstruction_error'] = mse
        results_df.to_csv('data/predictions.csv', index=False)
        print("Predictions saved to data/predictions.csv")

if __name__ == "__main__":
    train()