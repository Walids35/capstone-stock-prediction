from dataclasses import dataclass
from pathlib import Path

@dataclass
class ExpConfigs:
    model_name: str
    data_path: Path
    model_path: Path
    scaler_path: Path
    predictions_path: Path
    seq_length: int
    hidden_size: int
    num_layers: int
    dropout: float
    test_ratio: float
    lr: float
    batch_size: int
    epochs: int
    feature_columns: list
    target_column: str
    ticker: str
    news_model: str
    force_retrain: bool
    seed: int