#!/usr/bin/env python3
"""
TimesNet Training Script for Stock Prediction
Adapted from PatchTST training pipeline to work with TimesNet model
"""

from pathlib import Path
import os
from loguru import logger
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stock_prediction.modeling.TimesNet import Model as TimesNetModel

from stock_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from stock_prediction.dataset_pipeline import TimeSeriesDatasetPipeline
from utils.configs import configs
from utils.tools import Seeding
from utils.ExpConfigs import ExpConfigs


def create_timesnet_configs(
    seq_length: int = 30,
    pred_length: int = 1,
    input_size: int = 7,
    target_column: str = "Binary_Price",
    dropout: float = 0.1
) -> ExpConfigs:
    """Create configuration object for TimesNet model"""
    
    # Determine task type based on target column
    is_classification = target_column.lower() == "binary_price"
    task_name = "classification" if is_classification else "short_term_forecast"
    
    # Create configs object with all required parameters
    class TimesNetConfigs:
        def __init__(self):
            # Sequence parameters
            self.seq_len = seq_length
            self.pred_len = pred_length
            self.seq_len_max_irr = None
            self.pred_len_max_irr = None
            self.label_len = seq_length // 2  # Half of sequence length
            
            # Model architecture
            self.d_model = 64   # Embedding dimension
            self.e_layers = 2   # Number of encoder layers  
            self.d_ff = 128     # Feed-forward dimension
            self.dropout = dropout
            
            # Input/Output
            self.enc_in = input_size  # Number of input features
            self.c_out = 1  # Output dimension
            self.n_classes = 1 if is_classification else 1  # Output classes (1 for binary classification)
            
            # Task configuration
            self.task_name = task_name
            self.features = 'M'  # Multivariate
            
            # Embedding parameters
            self.embed = 'timeF'  # Time features embedding
            self.freq = 'm'  # Monthly frequency (expects 1 time feature)
    
    return TimesNetConfigs()


def main(
    data_path: Path = configs.data_path,
    model_path: Path = configs.model_path,
    scaler_path: Path = configs.scaler_path,
    seq_length: int = configs.seq_length,
    batch_size: int = configs.batch_size,
    epochs: int = configs.epochs,
    lr: float = configs.lr,
    dropout: float = configs.dropout,
    test_ratio: float = configs.test_ratio,
    feature_columns: list = configs.feature_columns,
    target_column: str = configs.target_column,
    force_retrain: bool = configs.force_retrain,
    seed: int = configs.seed
):
    """Main training function for TimesNet model"""
    
    # Set random seeds for reproducibility
    seed = Seeding(seed)
    seed.set()
    
    logger.info("=" * 80)
    logger.info("STARTING TIMESNET TRAINING")
    logger.info("=" * 80)
    
    # Determine task type
    is_classification = target_column.lower() == "binary_price"
    task_type = "classification" if is_classification else "short_term_forecast"
    logger.info(f"Task: {task_type} ({target_column})")
    
    # Set model path if not provided
    if model_path is None:
        model_name = f"timesnet_{target_column.lower()}_model.pth"
        model_path = MODELS_DIR / model_name
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    pipeline = TimeSeriesDatasetPipeline(
        data_path, seq_length=seq_length, test_ratio=test_ratio,
        feature_columns=feature_columns, target_column=target_column
    )
    pipeline.load_and_preprocess()
    pipeline.fit_scaler()
    pipeline.transform()
    X_train, y_train, X_test, y_test = pipeline.train_test_split()
    
    # Create validation split (10% of training data)
    val_ratio = 0.1
    n_train = len(X_train)
    split_idx = int(n_train * (1 - val_ratio))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Input shape: {X_train.shape}, Target shape: {y_train.shape}")
    
    # Save scaler
    pipeline.save_scaler(scaler_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get input dimensions
    input_size = X_train.shape[2]
    
    # Create TimesNet configuration
    logger.info("Creating TimesNet configuration...")
    timesnet_configs = create_timesnet_configs(
        seq_length=seq_length,
        pred_length=1,  # Single-step prediction
        input_size=input_size,
        target_column=target_column,
        dropout=dropout,
    )
    
    logger.info(f"Model parameters:")
    logger.info(f"  - Sequence length: {timesnet_configs.seq_len}")
    logger.info(f"  - Model dimension: {timesnet_configs.d_model}")
    logger.info(f"  - Encoder layers: {timesnet_configs.e_layers}")
    logger.info(f"  - Feed-forward dimension: {timesnet_configs.d_ff}")
    
    # Initialize or load model
    if os.path.exists(model_path) and not force_retrain:
        logger.info(f"Loading existing model from {model_path}")
        model = TimesNetModel(timesnet_configs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.info("Creating new TimesNet model...")
        model = TimesNetModel(timesnet_configs).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Setup loss function and optimizer
    if is_classification:
        criterion = torch.nn.BCEWithLogitsLoss()
        logger.info("Using BCEWithLogitsLoss for classification")
    else:
        criterion = torch.nn.MSELoss()
        logger.info("Using MSELoss for regression")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Prepare data tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    if is_classification:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (xb, yb) in enumerate(progress_bar):
            xb, yb = xb.to(device), yb.to(device)
            
            # Prepare inputs for TimesNet
            batch_size_curr, seq_len, n_features = xb.shape
            
            # Create time mark (simple time features)
            x_mark = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            x_mark = x_mark.repeat(batch_size_curr, 1, 1) / seq_len  # Normalize
            
            # Create dummy target for TimesNet forward pass
            if is_classification:
                y_dummy = torch.zeros((batch_size_curr, 1, n_features), device=device)
                y_class = yb.squeeze()  # [batch_size]
            else:
                y_dummy = torch.zeros((batch_size_curr, 1, n_features), device=device)
                y_class = None
            
            optimizer.zero_grad()
            
            # Forward pass through TimesNet
            if is_classification:
                outputs = model(x=xb, x_mark=x_mark, y_class=y_class)
                predictions = outputs["pred_class"]  # [batch_size, n_classes]
                
                # Calculate loss
                if predictions.shape[0] != yb.shape[0]:
                    predictions = predictions[:yb.shape[0]]
                
                if len(predictions.shape) == 1:
                    predictions = predictions.unsqueeze(1)
                
                loss = criterion(predictions, yb)
            else:
                # For regression
                y_mask = torch.ones_like(y_dummy, device=device)
                outputs = model(x=xb, x_mark=x_mark, y=y_dummy, y_mask=y_mask)
                predictions = outputs["pred"]  # [batch_size, pred_len, n_features]
                
                # Take mean across features for single output
                if predictions.shape[-1] > 1:
                    predictions = predictions.mean(dim=-1, keepdim=True)  # [batch_size, pred_len, 1]
                
                # Take last time step
                predictions = predictions[:, -1:, :]  # [batch_size, 1, 1]
                predictions = predictions.squeeze(-1)  # [batch_size, 1]
                
                loss = criterion(predictions, yb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{train_loss/train_batches:.4f}'
            })
        
        train_loss /= train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                
                batch_size_curr, seq_len, n_features = xb.shape
                
                # Create time mark
                x_mark = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                x_mark = x_mark.repeat(batch_size_curr, 1, 1) / seq_len
                
                if is_classification:
                    y_class = yb.squeeze()
                    outputs = model(x=xb, x_mark=x_mark, y_class=y_class)
                    predictions = outputs["pred_class"]
                    
                    if predictions.shape[0] != yb.shape[0]:
                        predictions = predictions[:yb.shape[0]]
                    
                    if len(predictions.shape) == 1:
                        predictions = predictions.unsqueeze(1)
                    
                    loss = criterion(predictions, yb)
                else:
                    y_dummy = torch.zeros((batch_size_curr, 1, n_features), device=device)
                    y_mask = torch.ones_like(y_dummy, device=device)
                    
                    outputs = model(x=xb, x_mark=x_mark, y=y_dummy, y_mask=y_mask)
                    predictions = outputs["pred"]
                    
                    if predictions.shape[-1] > 1:
                        predictions = predictions.mean(dim=-1, keepdim=True)
                    
                    predictions = predictions[:, -1:, :].squeeze(-1)
                    loss = criterion(predictions, yb)
                
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), model_path)
            logger.info(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Final model save
    torch.save(model.state_dict(), model_path)
    
    logger.success("🎉 Training completed!")
    logger.success(f"Model saved to: {model_path}")
    logger.success(f"Scaler saved to: {scaler_path}")
    logger.success(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()