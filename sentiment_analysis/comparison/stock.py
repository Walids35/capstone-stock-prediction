import pandas as pd
import re
from stock_prediction.config import PROJ_ROOT
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from venny4py.venny4py import *
from models.base import BaseSentimentComparison
from loguru import logger

class StockSentimentComparison(BaseSentimentComparison):
    def __init__(self, path, ticker):
        super().__init__(path, ticker)

    @staticmethod
    def _drop_text_duplicates(
        df: pd.DataFrame,
        text_col: str,
        *,
        normalise: bool = True,
        keep: str = "first",
        inplace: bool = False
    ) -> pd.DataFrame:
        work = df if inplace else df.copy()
        if normalise:
            def _canon(text: str) -> str:
                text = str(text).lower()
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"[^\w\s]", "", text)
                return text.strip()

            canon_col = f"__canon_{text_col}"
            work[canon_col] = work[text_col].map(_canon, na_action="ignore")
            deduped = work.drop_duplicates(subset=[canon_col], keep=keep)
        else:
            deduped = work.drop_duplicates(subset=[text_col], keep=keep)

        return deduped

    def load_fin_data(self, path):
        import re
        df = pd.read_csv(path)
        df["sentiment_label"] = pd.Categorical(
            df["sentiment_label"],
            categories=["Bearish", "Somewhat-Bearish", "Neutral", "Somewhat-Bullish", "Bullish"],
            ordered=True
        )
        df["title_sentiment_class"] = self.sentiment_to_numeric(df["sentiment_label"])
        df = self._drop_text_duplicates(df, "title")
        return df

    def create_sample_data(self):
        temp = self.load_fin_data(self.path)
        sample_data = {
            'title': temp['title'],
            'original_sentiment': temp['title_sentiment_class'],
            'sentiment_score': temp['sentiment_score'],
            'date': temp['time_published'],
            'relevance_score': temp['relevance_score']
        }
        return pd.DataFrame(sample_data)

    def sentiment_to_numeric(self, series) -> pd.Series:
        mapping = {
            "Bearish": 0,
            "Somewhat-Bearish": 0,
            "Neutral": 1,
            "Somewhat-Bullish": 2,
            "Bullish": 2
        }
        if pd.api.types.is_categorical_dtype(series):
            return series.astype(str).map(mapping).astype("Int8")
        else:
            return series.map(mapping).astype("Int8")

    def create_detailed_comparison_table(self, df):
        logger.info("\n" + "="*80)
        logger.success("DETAILED MODEL COMPARISON REPORT")
        logger.info("="*80)
        models = ['finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']
        model_names = ['FinBERT', 'RoBERTa', 'DeBERTa']
        for model, name in zip(models, model_names):
            logger.success(f"\n{name} Performance:")
            logger.info("-" * 30)
            accuracy = accuracy_score(df['original_sentiment'], df[model])
            logger.success(f"Overall Accuracy: {accuracy:.3f}")
            report = classification_report(df['original_sentiment'], df[model],
                                         target_names=['Negative', 'Neutral', 'Positive'],
                                         output_dict=True)
            logger.info(f"Precision - Negative: {report['Negative']['precision']:.3f}")
            logger.info(f"Precision - Neutral: {report['Neutral']['precision']:.3f}")
            logger.info(f"Precision - Positive: {report['Positive']['precision']:.3f}")

    def add_classical_model_predictions(self, df, models_dir="models/"):
        """
        Loads SVM, Logistic Regression, and Random Forest models from models_dir,
        predicts sentiment using x_cols, and adds predictions as new columns to df.
        Returns the updated dataframe.
        """
        import joblib
        import os
        x_cols = [
            "finbert_sentiment", "finbert_confidence",
            "roberta_sentiment", "roberta_confidence",
            "deberta_sentiment", "deberta_confidence"
        ]
        # Load models
        svm_path = os.path.join(models_dir, "svm_model.pkl")
        lr_path = os.path.join(models_dir, "logistic_regression.pkl")
        rf_path = os.path.join(models_dir, "random_forest_model.pkl")
        svm = joblib.load(svm_path)
        lr = joblib.load(lr_path)
        rf = joblib.load(rf_path)
        # Predict
        X = df[x_cols]
        logger.info(f"Predicting sentiment using {svm_path}, {lr_path}, {rf_path}")
        df["svm_sentiment"] = svm.predict(X)
        logger.info(f"Predicting sentiment using {lr_path}")
        df["lr_sentiment"] = lr.predict(X)
        logger.info(f"Predicting sentiment using {rf_path}")
        df["rf_sentiment"] = rf.predict(X)
        logger.info("Predictions complete")
        df.to_csv(f"{PROJ_ROOT}/data/interim/models/{self.ticker}_with_classical_models.csv", index=False)
        return df

    def create_comparison_table(self, df, models_dir="models/"):
        """
        Creates a comparison table of the sentiment predictions of the classical models.
        """
        logger.info("Creating comparison table")
        df = self.add_classical_model_predictions(df, models_dir)
        # Compute correlation matrix for the classical model predictions
        classical_cols = ["svm_sentiment", "lr_sentiment", "rf_sentiment","finbert_sentiment","roberta_sentiment","deberta_sentiment","original_sentiment"]
        correlation_matrix = df[classical_cols].corr()
        logger.info("Correlation matrix between classical model predictions:")
        logger.info(f"\n{correlation_matrix}")
        
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title(f"Correlation Matrix: Classical & Transformer Model Predictions ({self.ticker})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{PROJ_ROOT}/data/interim/models/{self.ticker}_classical_model_correlation_matrix.png")
        plt.close()
        return df
    