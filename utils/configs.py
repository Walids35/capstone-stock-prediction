import argparse
from dataclasses import dataclass
from pathlib import Path

from utils.ExpConfigs import ExpConfigs

parser = argparse.ArgumentParser(description="Stock Prediction")

# Model Paths
parser.add_argument("--model_name", type=str, default="lstm_model", help="Name of the model")
parser.add_argument("--model_path", type=str, default="models/lstm_model.pth", help="Path to the model file")

# Data Paths
parser.add_argument("--data_path", type=str, default="data/processed/AMZN_FINBERT_FLOAT.csv", help="Path to the data file")

# Model Hyperparameters
parser.add_argument("--scaler_path", type=str, default="models/lstm_model_norm.npz", help="Path to the scaler file")
parser.add_argument("--predictions_path", type=str, default="data/processed/test_predictions.csv", help="Path to the predictions file")
parser.add_argument("--seq_length", type=int, default=30, help="Sequence length")
parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")

configs = ExpConfigs(**vars(parser.parse_args()))