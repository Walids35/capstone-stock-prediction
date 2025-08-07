from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from loguru import logger
from stock_prediction.dataset_pipeline import TimeSeriesDatasetPipeline
import torch
from utils.configs import configs
from sklearn.metrics import roc_curve, roc_auc_score

class ProfitAnalyzer:
    def __init__(self, model, X_test, actual_prices, y_test, ticker, news_model, model_type="lstm", investment=1000.0, commission=0.001, risk_free_rate=0.02):
        self.model = model
        self.X_test = X_test
        self.actual_prices = actual_prices
        self.investment = investment
        self.commission = commission
        self.risk_free_rate = risk_free_rate
        self.y_test = y_test
        self.ticker = ticker
        self.news_model = news_model
        self.model_type = model_type
        # Metrics for export
        self.report = {}

    def model_evaluation(self, y_pred, y_test):
        # Ensure lengths match
        min_len = min(len(y_pred), len(y_test))
        y_pred = y_pred[:min_len]
        y_test = y_test[:min_len]
        
        logger.info(f"Model evaluation - y_pred length: {len(y_pred)}, y_test length: {len(y_test)}")
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        # Compute Youden's J statistic for optimal threshold
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        y_pred_binary = (y_pred > optimal_threshold).astype(int)

        return y_pred_binary, optimal_threshold

    def calculate_profit(self, buy_threshold=0, sell_threshold=0.42):
        if len(self.actual_prices) < len(self.X_test):
            raise ValueError("Length of actual_prices must match or exceed X_test.")

        # Make predictions with PyTorch model
        device = next(self.model.parameters()).device
        self.model.eval()
        
        batch_size = 32
        all_logits = []
        
        with torch.no_grad():
            for i in range(0, len(self.X_test), batch_size):
                batch_X = self.X_test[i:i+batch_size]
                batch_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
                
                # Handle different model types
                if hasattr(self.model, '__class__') and 'LSTM' in self.model.__class__.__name__:
                    # LSTM models don't need y parameter
                    model_output = self.model(batch_tensor)
                else:
                    # tPatchGNN, PatchTST, TimesNet need y parameter
                    batch_size_curr = batch_tensor.shape[0]
                    n_features = batch_tensor.shape[2]
                    y_dummy = torch.zeros((batch_size_curr, 1, n_features), device=device)
                    model_output = self.model(x=batch_tensor, y=y_dummy)
                if isinstance(model_output, dict):
                    # tPatchGNN, PatchTST, TimesNet return dictionaries
                    if "pred" in model_output:
                        batch_logits = model_output["pred"].cpu().numpy()
                    elif "pred_class" in model_output:
                        batch_logits = model_output["pred_class"].cpu().numpy()
                    else:
                        raise ValueError(f"Unknown model output format: {model_output.keys()}")
                else:
                    # LSTM and other models return tensor directly
                    batch_logits = model_output.cpu().numpy()
                
                all_logits.append(batch_logits)
    
        logits = np.concatenate(all_logits, axis=0).flatten()
        
        predictions = 1 / (1 + np.exp(-logits))
        pred_mean, pred_std = np.mean(predictions), np.std(predictions)
        pred_min, pred_max = np.min(predictions), np.max(predictions)
        logger.info(f"Prediction stats - Min: {pred_min:.4f}, Max: {pred_max:.4f}, Mean: {pred_mean:.4f}, Std: {pred_std:.4f}")
        logger.info(f"Lengths - predictions: {len(predictions)}, y_test: {len(self.y_test)}, X_test: {len(self.X_test)}")
        _, optimal_threshold = self.model_evaluation(predictions, self.y_test)
        logger.info(f"Optimal threshold: {optimal_threshold}")
        buy_threshold = optimal_threshold
        sell_threshold = optimal_threshold
        logger.info(f"Buy threshold: {buy_threshold}, Sell threshold: {sell_threshold}")

        balance = self.investment
        holding = False
        entry_price = 0
        profit_history = [balance]
        decisions = []
        daily_returns = []
        prev_balance = balance

        for i in range(len(predictions)):
            norm_pred = predictions[i]
            if i >= len(self.actual_prices):
                logger.warning(f"Index {i} out of bounds for actual_prices")
                break

            current_price = self.actual_prices[i]

            if norm_pred > buy_threshold and not holding:
                holding = True
                entry_price = current_price
                balance -= balance * self.commission
                decisions.append(1)
            elif norm_pred < sell_threshold and holding:
                balance *= current_price / entry_price
                balance -= balance * self.commission
                holding = False
                decisions.append(-1)
            else:
                decisions.append(0)

            profit_history.append(balance)
            daily_returns.append((balance / prev_balance) - 1)
            prev_balance = balance

        if holding and len(self.actual_prices) > 0:
            last_price = self.actual_prices[min(len(predictions)-1, len(self.actual_prices)-1)]
            balance *= last_price / entry_price
            balance -= balance * self.commission

        total_profit = balance - self.investment
        profit_pct = (balance / self.investment - 1) * 100
        mdd = self._calculate_max_drawdown(profit_history)
        sharpe_ratio, annual_return, annual_volatility = self._calculate_sharpe_ratio(daily_returns)
        calmar_ratio = annual_return / (mdd / 100) if mdd != 0 else 0

        logger.info(f"Initial investment: ${self.investment:.2f}")
        logger.info(f"Final balance: ${balance:.2f}")
        logger.info(f"Total profit: ${total_profit:.2f} ({profit_pct:.2f}%)")
        logger.info(f"Trades: {decisions.count(1)} buys, {decisions.count(-1)} sells")
        logger.info(f"Max Drawdown: {mdd:.2f}%")
        logger.info(f"Daily Volatility: {np.std(daily_returns):.4f}")
        logger.info(f"Annual Return: {annual_return:.2%}")
        logger.info(f"Annual Volatility: {annual_volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"Calmar Ratio: {calmar_ratio:.4f}")

        # Store metrics for export
        self.report = {
            "initial_investment": self.investment,
            "final_balance": balance,
            "total_profit": total_profit,
            "profit_percentage": profit_pct,
            "buy_count": decisions.count(1),
            "sell_count": decisions.count(-1),
            "max_drawdown": mdd,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio
        }

        return total_profit, profit_history, decisions

    def _calculate_max_drawdown(self, profits):
        cumulative_max = np.maximum.accumulate(profits)
        drawdowns = 1 - np.array(profits) / cumulative_max
        return np.max(drawdowns) * 100

    def _calculate_sharpe_ratio(self, daily_returns):
        period_risk_free = (1 + self.risk_free_rate) ** (1 / 252) - 1
        mean_return = np.mean(daily_returns)
        daily_vol = np.std(daily_returns)

        annual_return = ((1 + mean_return) ** 252) - 1
        annual_vol = daily_vol * np.sqrt(252)

        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol if annual_vol != 0 else 0
        return sharpe_ratio, annual_return, annual_vol

    def plot_profit(self, profit_history, decisions):
        min_len = min(len(profit_history), len(decisions)+1, len(self.actual_prices))
        profit_history = profit_history[:min_len]
        decisions = decisions[:min_len-1]
        prices = self.actual_prices[:min_len-1]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(profit_history, 'b-', linewidth=2)
        plt.title('Profit History')
        plt.ylabel('Balance ($)')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(prices, 'k-', linewidth=1)

        buy_idx = [i for i, d in enumerate(decisions) if d == 1]
        sell_idx = [i for i, d in enumerate(decisions) if d == -1]

        plt.plot(buy_idx, [prices[i] for i in buy_idx], '^', color='g', label='Buy')
        plt.plot(sell_idx, [prices[i] for i in sell_idx], 'v', color='r', label='Sell')

        plt.title('Trading Decisions')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"reports/simulation/{self.ticker}_{self.model_type}_Binary_Price_{self.news_model}_trading_decisions.png")

    def plot_threshold_heatmap(self, threshold_results):
        try:
            thresholds = sorted(set([t[0] for t in threshold_results.keys()] + [t[1] for t in threshold_results.keys()]))
            profits = np.zeros((len(thresholds), len(thresholds)))

            for (buy_t, sell_t), profit in threshold_results.items():
                i, j = thresholds.index(buy_t), thresholds.index(sell_t)
                profits[i, j] = profit

            plt.figure(figsize=(10, 8))
            sns.heatmap(profits, annot=True, fmt=".1f", cmap="viridis",
                        xticklabels=thresholds, yticklabels=thresholds)
            plt.xlabel('Sell Threshold')
            plt.ylabel('Buy Threshold')
            plt.title('Profit by Buy/Sell Threshold')
            plt.savefig(f"reports/simulation/{self.ticker}_{self.model_type}_Binary_Price_{self.news_model}_trading_threshold_heatmap.png")
        except Exception as e:
            logger.error(f"Heatmap failed: {e}")
            best = max(threshold_results.items(), key=lambda x: x[1])
            logger.info(f"Best thresholds: Buy={best[0][0]}, Sell={best[0][1]} with profit ${best[1]:.2f}")

    def export_report(self, filepath):
        if not self.report:
            logger.warning("No report to export. Run `calculate_profit()` first.")
            return

        try:
            if filepath.endswith(".json"):
                with open(filepath, 'w') as f:
                    json.dump(self.report, f, indent=4)
            elif filepath.endswith(".csv"):
                df = pd.DataFrame([self.report])
                df.to_csv(filepath, index=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")

            logger.success(f"Performance report exported to: {filepath}")
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")

    def run(self):
        try:
            profit, history, decisions = self.calculate_profit()
            self.plot_profit(history, decisions)
            return profit, history, decisions
        except Exception as e:
            logger.error(f"Run error: {e}")
            logger.debug(f"X_test shape: {self.X_test.shape}, actual_prices length: {len(self.actual_prices)}")
            return None, None, None


def main(
    data_path: Path = configs.data_path, 
    seq_length: int = configs.seq_length, 
    test_ratio: float = configs.test_ratio, 
    feature_columns: list = configs.feature_columns, 
    target_column: str = configs.target_column, 
    scaler_path: Path = configs.scaler_path, 
    model_path: Path = configs.model_path,
    ticker: str = configs.ticker,
    news_model: str = configs.news_model,
    batch_size: int = configs.batch_size,
    dropout: float = configs.dropout,
):

    pipeline = TimeSeriesDatasetPipeline(
        data_path, seq_length=seq_length, test_ratio=test_ratio,
        feature_columns=feature_columns, target_column=target_column
    )
    pipeline.load_and_preprocess()
    pipeline.load_scaler(scaler_path)
    pipeline.transform()
    X_train, y_train, X_test, y_test = pipeline.train_test_split()
    input_size = X_test.shape[2]

    # Get actual prices from the test set by using the pipeline's inverse transform method
    # We need to get the first feature from each sequence and inverse transform it properly
    first_feature_values = X_test[:, 0, 0]  # Get first feature from first timestep of each sequence
    actual_prices = pipeline.inverse_transform_features(first_feature_values, feature_idx=0)
    
    # Load model properly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(model_path, map_location=device)
    
    # Determine model type from model path
    model_filename = str(model_path)
    if "tpatchgnn" in model_filename:
        from stock_prediction.modeling.tPatchGNN import Model
        from stock_prediction.modeling.train_tpatchgnn import create_tpatchgnn_configs
        model_configs = create_tpatchgnn_configs(
            seq_length=seq_length,
            pred_length=1,
            input_size=input_size,
            target_column=target_column,
            dropout=dropout,
            batch_size=batch_size
        )
        model = Model(model_configs).to(device)
    elif "patchtst" in model_filename:
        from stock_prediction.modeling.PatchTST import Model
        from stock_prediction.modeling.train_patchtst import create_patchtst_configs
        model_configs = create_patchtst_configs(
            seq_length=seq_length,
            pred_length=1,
            input_size=input_size,
            target_column=target_column,
            dropout=dropout
        )
        model = Model(model_configs).to(device)
    elif "timesnet" in model_filename:
        from stock_prediction.modeling.TimesNet import Model
        from stock_prediction.modeling.train_timesnet import create_timesnet_configs
        model_configs = create_timesnet_configs(
            seq_length=seq_length,
            pred_length=1,
            input_size=input_size,
            target_column=target_column,
            dropout=dropout
        )
        model = Model(model_configs).to(device)
    elif "lstm" in model_filename:
        from stock_prediction.modeling.LSTM import LSTMModel
        output_size = 1 if target_column.lower() != "binary_price" else 1
        model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2, 
            output_size=output_size,
            dropout=dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown model type in filename: {model_filename}")
    
    # Load the state dict
    model.load_state_dict(model_state_dict)
    model.eval()

    # Determine model type for file naming
    model_type = "lstm"  # default
    if "tpatchgnn" in model_filename:
        model_type = "tpatchgnn"
    elif "patchtst" in model_filename:
        model_type = "patchtst"
    elif "timesnet" in model_filename:
        model_type = "timesnet"
    
    analyzer = ProfitAnalyzer(model, X_test, actual_prices, y_test, ticker, news_model, model_type)
    profit, history, decisions = analyzer.run()    
    analyzer.export_report(f"reports/simulation/{ticker}_{model_type}_Binary_Price_{news_model}_trading_analysis.json")


if __name__ == "__main__":
    main()