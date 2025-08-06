from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
import torch
from stock_prediction.modeling.LSTM import LSTMModel
from stock_prediction.dataset_pipeline import TimeSeriesDatasetPipeline
from utils.configs import configs
from utils.metrics import metrics
import matplotlib.pyplot as plt
from utils.tools import Seeding
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report, roc_auc_score

def model_evaluation(y_pred, y_test, ticker, news_model, target_column):

  # Compute ROC curve and AUC
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred)

  # Compute Youden's J statistic for optimal threshold
  j_scores = tpr - fpr
  optimal_idx = np.argmax(j_scores)
  optimal_threshold = thresholds[optimal_idx]

  y_pred_binary = (y_pred > optimal_threshold).astype(int)

  # Plot ROC curve
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
  plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

  # Highlight optimal point
  plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.4f}')
  plt.text(fpr[optimal_idx]+0.02, tpr[optimal_idx]-0.05, f'Th={optimal_threshold:.2f}', color='red')

  # Styling
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.xlabel('False Positive Rate (FPR)')
  plt.ylabel('True Positive Rate (TPR)')
  plt.legend(loc='lower right')
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f"reports/output/LSTM_wo_count_sum/{ticker}_{news_model}_{target_column}_roc_curve.png")

  return y_pred_binary, optimal_threshold

def main(
    data_path: Path = configs.data_path,
    model_path: Path = configs.model_path,
    scaler_path: Path = configs.scaler_path,
    seq_length: int = configs.seq_length,
    hidden_size: int = configs.hidden_size,
    num_layers: int = configs.num_layers,
    dropout: float = configs.dropout,
    test_ratio: float = configs.test_ratio,
    feature_columns: list = configs.feature_columns,
    target_column: str = configs.target_column,
    ticker: str = configs.ticker,
    news_model: str = configs.news_model,
    seed: int = configs.seed
):
    seed = Seeding(seed)
    seed.set()
    
    pipeline = TimeSeriesDatasetPipeline(
        data_path, seq_length=seq_length, test_ratio=test_ratio,
        feature_columns=feature_columns, target_column=target_column
    )
    pipeline.load_and_preprocess()
    pipeline.load_scaler(scaler_path)
    pipeline.transform()
    X_train, y_train, X_test, y_test = pipeline.train_test_split()
    
    # Account for validation split that was created during training
    # During training, 10% of training data was used for validation (match Keras)
    val_ratio = 0.1
    n_train = len(X_train)
    split_idx = int(n_train * (1 - val_ratio))
    # Use the same split as during training (only test set for prediction)
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]
    
    logger.info(f"Data splits for prediction - Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Test set shape: {X_test.shape}, Target shape: {y_test.shape}")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Classification mode check ---
    is_classification = target_column.lower() == "binary_price"

    output_size = 1 
    model = LSTMModel(
        input_size=X_test.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Add debugging information
    logger.info(f"Model loaded successfully. Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Input shape for prediction: {X_test.shape}")
    
    # Check for potential data issues
    logger.info(f"X_test stats - Min: {X_test.min():.4f}, Max: {X_test.max():.4f}, Mean: {X_test.mean():.4f}, Std: {X_test.std():.4f}")
    logger.info(f"y_test stats - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")

    logger.info("Making predictions with LSTM model...")
    
    # Process in batches to handle BatchNorm properly
    batch_size = 32
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
            batch_logits = model(batch_tensor).cpu().numpy()
            all_logits.append(batch_logits)
    
    logits = np.concatenate(all_logits, axis=0).flatten()
    logger.info(f"Raw logits stats - Min: {logits.min():.4f}, Max: {logits.max():.4f}, Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    
    # Check for constant predictions
    unique_values = len(np.unique(np.round(logits, 4)))
    logger.info(f"Number of unique prediction values (rounded to 4 decimals): {unique_values}")
    
    if np.std(logits) < 1e-6:
        logger.warning("⚠️  Model is predicting nearly constant values! This indicates a learning problem.")
        logger.warning(f"Logits standard deviation: {np.std(logits):.8f}")
        logger.warning("Consider retraining with force_retrain=True")
    elif unique_values < 10:
        logger.warning(f"⚠️  Model is predicting only {unique_values} unique values. This may indicate poor learning.")
        logger.warning("Consider retraining with force_retrain=True or adjusting hyperparameters.")

    # --- Post-process predictions ---
    if is_classification:
        probs = 1 / (1 + np.exp(-logits))
        y_true = y_test.astype(int)

        # 2. Evaluate model: plot ROC & find Youden’s‐optimal threshold
        preds, optimal_threshold = model_evaluation(probs, y_true, ticker, news_model, target_column)

        # 3. Compute additional metrics
        accuracy    = accuracy_score(y_true, preds)
        auc_score   = roc_auc_score(y_true, probs)
        conf_matrix = confusion_matrix(y_true, preds).tolist()
        class_report= classification_report(y_true, preds, output_dict=True)

        # 4. Bundle results
        results = {
            "optimal_threshold": float(optimal_threshold),
            "auc":               float(auc_score),
            "accuracy":          float(accuracy),
            "confusion_matrix":  conf_matrix,
            "classification_report": class_report
        }
    else:
        preds = pipeline.inverse_transform_targets(logits)
        y_true = pipeline.inverse_transform_targets(y_test)
        results = metrics(preds, y_true)

    logger.info(f"Metrics: {results}")


    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True Values", alpha=0.7)
    plt.plot(preds, label="Predicted Values", alpha=0.7)
    plt.title("True vs Predicted Values")
    plt.xlabel("Time Step")
    plt.ylabel("Value" if not is_classification else "Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/output/LSTM_wo_count_sum/{ticker}_{news_model}_{target_column}_pred_vs_true.png")

    # --- Save run info ---
    with open(f"reports/output/LSTM_wo_count_sum/{ticker}_{news_model}_{target_column}_pred_vs_true.txt", "w") as f:
        f.write(f"Date: {datetime.now()} \n")
        f.write(f"Ticker: {ticker} \n")
        f.write(f"Feature Columns: {feature_columns} \n")
        f.write(f"Target Column: {target_column} \n")
        f.write(f"Sequence Length: {seq_length} \n")
        f.write(f"Hidden Size: {hidden_size} \n")
        f.write(f"Number of Layers: {num_layers} \n")
        f.write(f"Dropout: {dropout} \n")
        f.write(f"Metrics: {results} \n")



if __name__ == "__main__":
    main()
