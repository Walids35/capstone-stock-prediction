from pathlib import Path
import os
from loguru import logger
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stock_prediction.modeling.LSTM import LSTMModel

from stock_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from stock_prediction.dataset_pipeline import TimeSeriesDatasetPipeline
from utils.configs import configs
from utils.tools import Seeding


def main(
    data_path: Path = configs.data_path,
    model_path: Path = configs.model_path,
    scaler_path: Path = configs.scaler_path,
    seq_length: int = configs.seq_length,
    batch_size: int = 32,
    epochs: int = configs.epochs,
    lr: float = configs.lr,
    hidden_size: int = configs.hidden_size,
    num_layers: int = configs.num_layers,
    dropout: float = configs.dropout,
    test_ratio: float = configs.test_ratio,
    feature_columns: list = configs.feature_columns,
    target_column: str = configs.target_column,
    force_retrain: bool = configs.force_retrain,
):
    seed = Seeding(812)
    seed.set()

    is_classification = target_column.lower() == "binary_price"

    pipeline = TimeSeriesDatasetPipeline(
        data_path, seq_length=seq_length, test_ratio=test_ratio,
        feature_columns=feature_columns, target_column=target_column
    )
    pipeline.load_and_preprocess()
    pipeline.fit_scaler()
    pipeline.transform()
    X_train, y_train, X_test, y_test = pipeline.train_test_split()
    
    # Create validation split from training data (match Keras 10%)
    val_ratio = 0.1
    n_train = len(X_train)
    split_idx = int(n_train * (1 - val_ratio))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    pipeline.save_scaler(scaler_path)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    input_size = X_train.shape[2]
    output_size = 1

    # Load or initialize model
    if os.path.exists(configs.model_path) and not force_retrain:
        logger.info(f"Loading existing model from {configs.model_path}")
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout).to(device)
        model.load_state_dict(torch.load(configs.model_path, map_location=device))
    else:
        logger.info(f"Training new model from scratch")
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout).to(device)

    # Choose appropriate loss function
    if is_classification:
        criterion = torch.nn.BCEWithLogitsLoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    else:
        criterion = torch.nn.MSELoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Training LSTM model...")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        
        val_loss /= len(val_dataset)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}")
    logger.success(f"Scaler params saved to {scaler_path}")


if __name__ == "__main__":
    main()
