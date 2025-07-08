import numpy as np
import pandas as pd
from pathlib import Path

class TimeSeriesDatasetPipeline:
    def __init__(self, csv_path, seq_length=30, test_ratio=0.2):
        self.csv_path = Path(csv_path)
        self.seq_length = seq_length
        self.test_ratio = test_ratio
        self.scaler_params = {}
        self.X_train = self.y_train = self.X_test = self.y_test = None

    def load_and_preprocess(self):
        df = pd.read_csv(self.csv_path)
        # Drop date column (assume it's the first column)
        df = df.iloc[:, 1:]
        # Last column is target
        features = df.iloc[:, :-1].values.astype(np.float32)
        targets = df.iloc[:, -1].values.astype(np.float32)
        self.features = features
        self.targets = targets
        return features, targets

    def fit_scaler(self):
        feat_min = self.features.min(axis=0)
        feat_max = self.features.max(axis=0)
        target_min = self.targets.min()
        target_max = self.targets.max()
        self.scaler_params = {
            'feat_min': feat_min,
            'feat_max': feat_max,
            'target_min': target_min,
            'target_max': target_max
        }
        return self.scaler_params

    def transform(self):
        p = self.scaler_params
        features_norm = (self.features - p['feat_min']) / (p['feat_max'] - p['feat_min'] + 1e-8)
        targets_norm = (self.targets - p['target_min']) / (p['target_max'] - p['target_min'] + 1e-8)
        self.features_norm = features_norm
        self.targets_norm = targets_norm
        return features_norm, targets_norm

    def inverse_transform_targets(self, targets_norm):
        p = self.scaler_params
        return targets_norm * (p['target_max'] - p['target_min'] + 1e-8) + p['target_min']

    def make_sequences(self, features, targets):
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(targets[i+self.seq_length])
        return np.array(X), np.array(y)

    def train_test_split(self):
        n = len(self.features_norm) - self.seq_length
        split = int(n * (1 - self.test_ratio))
        X, y = self.make_sequences(self.features_norm, self.targets_norm)
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_test, self.y_test = X[split:], y[split:]
        return self.X_train, self.y_train, self.X_test, self.y_test

    def save_scaler(self, path):
        np.savez(path, **self.scaler_params)

    def load_scaler(self, path):
        d = np.load(path)
        self.scaler_params = {k: d[k] for k in d}
        return self.scaler_params 