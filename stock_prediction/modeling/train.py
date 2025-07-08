from pathlib import Path

from loguru import logger
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stock_prediction.modeling.lstm_model import LSTMModel

from stock_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from stock_prediction.dataset_pipeline import TimeSeriesDatasetPipeline
from utils.configs import configs


def main(
    data_path: Path = configs.data_path,
    model_path: Path = configs.model_path,
    scaler_path: Path = configs.scaler_path,
    seq_length: int = configs.seq_length,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = configs.lr,
    hidden_size: int = configs.hidden_size,
    num_layers: int = configs.num_layers,
    dropout: float = configs.dropout,
    test_ratio: float = configs.test_ratio,
):
    pipeline = TimeSeriesDatasetPipeline(data_path, seq_length=seq_length, test_ratio=test_ratio)
    pipeline.load_and_preprocess()
    pipeline.fit_scaler()
    pipeline.transform()
    X_train, y_train, X_test, y_test = pipeline.train_test_split()
    pipeline.save_scaler(scaler_path)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model if it exists
    if Path(configs.model_path).exists():
        logger.info(f"Loading model from {configs.model_path}")
        model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
        model.load_state_dict(torch.load(configs.model_path, map_location=device))
    else:
        logger.info(f"Training new model")
        model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info("Training LSTM model...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.4f}")
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}")
    logger.success(f"Scaler params saved to {scaler_path}")


if __name__ == "__main__":
    main()
