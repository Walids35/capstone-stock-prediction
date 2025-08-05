#!/usr/bin/env python3
"""
tPatchGNN Prediction Script for Stock Prediction
Adapted from PatchTST prediction pipeline to work with tPatchGNN model
"""

from pathlib import Path
import os
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from stock_prediction.modeling.tPatchGNN import Model as tPatchGNNModel
from stock_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from stock_prediction.dataset_pipeline import TimeSeriesDatasetPipeline
from utils.configs import configs
from utils.tools import Seeding
from utils.ExpConfigs import ExpConfigs
from utils.metrics import RSE, CORR, MAE, MAPE, RMSE, MSLE


def create_tpatchgnn_configs(
    seq_length: int = 30,
    pred_length: int = 1,
    input_size: int = 7,
    target_column: str = "Binary_Price",
    dropout: float = 0.1,
    batch_size: int = 32
) -> ExpConfigs:
    """Create configuration object for tPatchGNN model"""
    
    # Determine task type based on target column
    is_classification = target_column.lower() == "binary_price"
    task_name = "classification" if is_classification else "short_term_forecast"
    
    # Create configs object with all required parameters
    class tPatchGNNConfigs:
        def __init__(self):
            # Sequence parameters
            self.seq_len = seq_length
            self.pred_len = pred_length
            self.seq_len_max_irr = None
            self.pred_len_max_irr = None
            
            # Patch parameters
            self.patch_len = 10  # Length of each patch (must divide seq_len)
            while seq_length % self.patch_len != 0:
                self.patch_len -= 1
                if self.patch_len < 2:
                    self.patch_len = seq_length  # fallback
                    break
            
            # Input/Output
            self.enc_in = input_size  # Number of input features
            self.n_classes = 2 if is_classification else 1  # Output classes
            
            # Task configuration
            self.task_name = task_name
            self.features = 'M'  # Multivariate
            
            # tPatchGNN specific parameters
            self.batch_size = batch_size  # Required for model construction
            self.tpatchgnn_te_dim = 16   # Time embedding dimension
            self.node_dim = 32           # Node dimension for GNN
            self.n_heads = 4             # Number of attention heads
            self.dropout = dropout
    
    return tPatchGNNConfigs()


def model_evaluation(y_pred, y_test, ticker, news_model, target_column):
    """Evaluate model predictions and save results"""
    
    is_classification = target_column.lower() == "binary_price"
    
    if is_classification:
        # Binary classification evaluation
        y_pred_proba = torch.sigmoid(torch.tensor(y_pred)).numpy()
        
        # Find optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Apply threshold
        y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        class_report = classification_report(y_test, y_pred_binary, output_dict=True)
        
        metrics = {
            'optimal_threshold': optimal_threshold,
            'auc': roc_auc,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        logger.info(f"Classification Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC: {roc_auc:.4f}")
        logger.info(f"  Optimal Threshold: {optimal_threshold:.4f}")
        
        # Save ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {ticker} {news_model} {target_column}')
        plt.legend(loc="lower right")
        
        roc_path = REPORTS_DIR / "output" / "tPatchGNN" / f"{ticker}_{news_model}_{target_column}_roc_curve.png"
        roc_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return y_pred_binary, optimal_threshold, metrics
    
    else:
        # Regression evaluation
        metrics = {
            'RSE': RSE(y_pred, y_test),
            'CORR': CORR(y_pred, y_test),
            'MAE': MAE(y_pred, y_test),
            'MAPE': MAPE(y_pred, y_test),
            'RMSE': RMSE(y_pred, y_test),
            'MSLE': MSLE(y_pred, y_test)
        }
        
        logger.info(f"Regression Results:")
        logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"  MAE: {metrics['MAE']:.4f}")
        logger.info(f"  CORR: {metrics['CORR']:.4f}")
        
        return y_pred, None, metrics


def main(
    data_path: Path = configs.data_path,
    model_path: Path = configs.model_path,
    scaler_path: Path = configs.scaler_path,
    seq_length: int = configs.seq_length,
    dropout: float = configs.dropout,
    test_ratio: float = configs.test_ratio,
    feature_columns: list = configs.feature_columns,
    target_column: str = configs.target_column,
    ticker: str = configs.ticker,
    news_model: str = configs.news_model,
):
    """Main prediction function for tPatchGNN model"""
    
    # Set random seeds for reproducibility
    seed = Seeding(812)
    seed.set()
    
    logger.info("=" * 80)
    logger.info("STARTING tPatchGNN PREDICTION")
    logger.info("=" * 80)
    
    # Determine task type
    is_classification = target_column.lower() == "binary_price"
    task_type = "Classification" if is_classification else "Regression"
    logger.info(f"Task: {task_type} ({target_column})")
    logger.info(f"Ticker: {ticker}, News Model: {news_model}")
    
    # Set model path if not provided
    if model_path is None:
        model_name = f"tpatchgnn_{target_column.lower()}_model.pth"
        model_path = MODELS_DIR / model_name
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first using train_tpatchgnn.py")
        return
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    pipeline = TimeSeriesDatasetPipeline(
        data_path, seq_length=seq_length, test_ratio=test_ratio,
        feature_columns=feature_columns, target_column=target_column
    )
    pipeline.load_and_preprocess()
    pipeline.load_scaler(scaler_path)
    pipeline.transform()
    X_train, y_train, X_test, y_test = pipeline.train_test_split()
    
    # Account for validation split that was created during training
    val_ratio = 0.1
    n_train = len(X_train)
    split_idx = int(n_train * (1 - val_ratio))
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]
    
    logger.info(f"Data splits for prediction - Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Test set shape: {X_test.shape}, Target shape: {y_test.shape}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get input dimensions
    input_size = X_test.shape[2]
    
    # Create tPatchGNN configuration
    logger.info("Creating tPatchGNN configuration...")
    batch_size = 32  # Use same batch size as training
    tpatchgnn_configs = create_tpatchgnn_configs(
        seq_length=seq_length,
        pred_length=1,
        input_size=input_size,
        target_column=target_column,
        dropout=dropout,
        batch_size=batch_size
    )
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tPatchGNNModel(tpatchgnn_configs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Prepare test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Run inference
    logger.info("Running inference...")
    all_predictions = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        num_batches = (len(X_test) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_test))
            
            xb = X_test_tensor[start_idx:end_idx]
            batch_size_curr, seq_len, n_features = xb.shape
            
            # Prepare inputs for tPatchGNN
            y_dummy = torch.zeros((batch_size_curr, 1, n_features), device=device)
            
            outputs = model(x=xb, y=y_dummy)
            predictions = outputs["pred"]  # [batch_size, pred_len, n_features] or [batch_size, 1, 1] for classification
            
            # Process predictions based on task type
            if is_classification:
                # For classification, tPatchGNN returns [batch_size, 1, 1]
                predictions = predictions.squeeze(-1)  # [batch_size, 1]
            else:
                # For regression, take mean across features and time steps
                if predictions.shape[-1] > 1:
                    predictions = predictions.mean(dim=-1, keepdim=True)  # [batch_size, pred_len, 1]
                predictions = predictions[:, -1:, :].squeeze(-1)  # [batch_size, 1]
            
            batch_pred = predictions.cpu().numpy()
            
            all_predictions.extend(batch_pred.flatten())
    
    y_pred = np.array(all_predictions)
    
    logger.info(f"Predictions shape: {y_pred.shape}")
    logger.info(f"Sample predictions: {y_pred[:5]}")
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred_processed, optimal_threshold, metrics = model_evaluation(
        y_pred, y_test, ticker, news_model, target_column
    )
    
    # Save results
    output_dir = REPORTS_DIR / "output" / "tPatchGNN"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to text file
    results_file = output_dir / f"{ticker}_{news_model}_{target_column}_pred_vs_true.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"Date: {pd.Timestamp.now()} \n")
        f.write(f"Ticker: {ticker} \n")
        f.write(f"Feature Columns: {feature_columns} \n")
        f.write(f"Target Column: {target_column} \n")
        f.write(f"Sequence Length: {seq_length} \n")
        f.write(f"Model: tPatchGNN \n")
        f.write(f"Patch Length: {tpatchgnn_configs.patch_len} \n")
        f.write(f"Time Embedding Dim: {tpatchgnn_configs.tpatchgnn_te_dim} \n")
        f.write(f"Node Dimension: {tpatchgnn_configs.node_dim} \n")
        f.write(f"Attention Heads: {tpatchgnn_configs.n_heads} \n")
        f.write(f"Dropout: {dropout} \n")
        f.write(f"Metrics: {metrics} \n")
    
    # Create prediction vs true plot
    plt.figure(figsize=(12, 8))
    
    if is_classification:
        # For classification, plot probabilities
        y_pred_proba = torch.sigmoid(torch.tensor(y_pred)).numpy()
        
        plt.subplot(2, 1, 1)
        plt.scatter(range(len(y_test)), y_test, alpha=0.6, label='True', s=20)
        plt.scatter(range(len(y_pred_proba)), y_pred_proba, alpha=0.6, label='Predicted Probability', s=20)
        plt.axhline(y=optimal_threshold, color='red', linestyle='--', label=f'Threshold ({optimal_threshold:.3f})')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'{ticker} {news_model} {target_column} - tPatchGNN Predictions vs True')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.scatter(range(len(y_test)), y_test, alpha=0.6, label='True', s=20)
        plt.scatter(range(len(y_pred_processed)), y_pred_processed, alpha=0.6, label='Predicted Binary', s=20)
        plt.xlabel('Sample Index')
        plt.ylabel('Binary Value')
        plt.title('Binary Predictions after Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # For regression, plot continuous values as lines
        plt.plot(range(len(y_test)), y_test, label='True', linewidth=2, alpha=0.8)
        plt.plot(range(len(y_pred)), y_pred, label='Predicted', linewidth=2, alpha=0.8)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'{ticker} {news_model} {target_column} - tPatchGNN Predictions vs True')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / f"{ticker}_{news_model}_{target_column}_pred_vs_true.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success("🎉 Prediction completed!")
    logger.success(f"Results saved to: {results_file}")
    logger.success(f"Plot saved to: {plot_path}")
    
    if is_classification:
        logger.success(f"Classification accuracy: {metrics['accuracy']:.4f}")
        logger.success(f"AUC: {metrics['auc']:.4f}")
    else:
        logger.success(f"RMSE: {metrics['RMSE']:.4f}")
        logger.success(f"Correlation: {metrics['CORR']:.4f}")


if __name__ == "__main__":
    main()