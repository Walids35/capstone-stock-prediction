import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

class TimeSeriesDatasetPipeline:
    def __init__(self, csv_path, seq_length=30, test_ratio=0.2, feature_columns=None, target_column=None):
        """
        Args:
            csv_path (str or Path): Path to the CSV file.
            seq_length (int): Sequence length for time series.
            test_ratio (float): Ratio of test set.
            feature_columns (list of str): List of column names to use as features.
            target_column (str): Name of the column to use as the target.
        """
        self.csv_path = Path(csv_path)
        self.seq_length = seq_length
        self.test_ratio = test_ratio
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_params = {}
        self.X_train = self.y_train = self.X_test = self.y_test = None

    def load_and_preprocess(self):
        df = pd.read_csv(self.csv_path)
        df = df.dropna()
        # If feature_columns or target_column not set, use all except date columns
        if self.feature_columns is None:
            # Exclude likely date columns and target columns
            exclude = {'Date', 'date_only', 'Float_Price', 'Binary_Price', 'Factor_Price', 'Delta_Price'}
            exclude = exclude.union({self.target_column} if self.target_column else set())
            self.feature_columns = [col for col in df.columns if col not in exclude]
        if self.target_column is None:
            # Default to last column
            self.target_column = df.columns[-1]
        # Select features and target
        features = df[self.feature_columns].values.astype(np.float32)
        targets = df[self.target_column].values.astype(np.float32)
        self.features = features
        self.targets = targets
        return features, targets

    def fit_scaler(self):
        # Combine features and targets for unified scaling like Keras
        combined_data = np.column_stack([self.features, self.targets.reshape(-1, 1)])
        self.scaler.fit(combined_data)
        
        # Store for compatibility
        self.scaler_params = {
            'scaler': self.scaler
        }
        return self.scaler_params

    def transform(self):
        # Use sklearn scaler like Keras implementation
        combined_data = np.column_stack([self.features, self.targets.reshape(-1, 1)])
        scaled_data = self.scaler.transform(combined_data)
        
        # Split back into features and targets
        self.features_norm = scaled_data[:, :-1]
        self.targets_norm = scaled_data[:, -1]
        
        return self.features_norm, self.targets_norm

    def inverse_transform_targets(self, targets_norm):
        # Create dummy array with zeros for features and actual target values
        n_features = self.features.shape[1]
        dummy_data = np.zeros((len(targets_norm), n_features + 1))
        dummy_data[:, -1] = targets_norm.flatten()
        
        # Inverse transform and extract target column
        inverse_data = self.scaler.inverse_transform(dummy_data)
        return inverse_data[:, -1]

    def inverse_transform_features(self, features_norm, feature_idx=0):
        # Create dummy array with zeros for features and actual feature values
        n_features = self.features.shape[1]
        dummy_data = np.zeros((len(features_norm), n_features + 1))
        dummy_data[:, feature_idx] = features_norm.flatten()
        
        # Inverse transform and extract feature column
        inverse_data = self.scaler.inverse_transform(dummy_data)
        return inverse_data[:, feature_idx]

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
        # Save sklearn scaler using pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, path):
        # Load sklearn scaler using pickle
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.scaler_params = {'scaler': self.scaler}
        return self.scaler_params 