from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
import torch
from stock_prediction.modeling.lstm_model import LSTMModel
from stock_prediction.dataset_pipeline import TimeSeriesDatasetPipeline

from stock_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from utils.configs import configs
from utils.metrics import metrics
import matplotlib.pyplot as plt

def main(
    data_path: Path = configs.data_path,
    model_path: Path = configs.model_path,
    scaler_path: Path = configs.scaler_path,
    predictions_path: Path = configs.predictions_path,
    seq_length: int = configs.seq_length,
    hidden_size: int = configs.hidden_size,
    num_layers: int = configs.num_layers,
    dropout: float = configs.dropout,
    test_ratio: float = configs.test_ratio,
):
    pipeline = TimeSeriesDatasetPipeline(data_path, seq_length=seq_length, test_ratio=test_ratio)
    pipeline.load_and_preprocess()
    pipeline.load_scaler(scaler_path)
    pipeline.transform()
    X_train, y_train, X_test, y_test = pipeline.train_test_split()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=X_test.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logger.info("Making predictions with LSTM model...")
    with torch.no_grad():
        preds_norm = model(torch.tensor(X_test).to(device)).cpu().numpy().flatten()
    preds = pipeline.inverse_transform_targets(preds_norm)

    # Calculate and log metrics
    y_true = pipeline.inverse_transform_targets(y_test)
    results = metrics(preds, y_true)
    logger.info(f"Metrics: {results}")

    # Plot true vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True Values")
    plt.plot(preds, label="Predicted Values")
    plt.title("True vs Predicted Values")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/pred_vs_true.png")
    plt.show()

    # Save predictions
    pred_df = pd.read_csv(data_path).iloc[-len(preds):].copy()
    pred_df["LSTM_Prediction"] = preds
    pred_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
