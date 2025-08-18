import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline
from venny4py.venny4py import *
from loguru import logger
import ast
from collections import Counter
from models.base import BaseSentimentComparison

# Set style for better plots
sns.set_palette("husl")

class FinancialSentimentComparison(BaseSentimentComparison):
    PRIORITY = ("positive", "negative", "neutral")

    def __init__(self, path, ticker):
        super().__init__(path, ticker)

    def sentiment_to_numeric(self, series) -> pd.Series:
        mapping = {"negative": 0, "positive": 2, "neutral": 1}
        if pd.api.types.is_categorical_dtype(series):
            return series.astype(str).map(mapping).astype("Int8")
        else:
            return series.map(mapping).astype("Int8")

    def single_sentiment(self, json_like: str) -> str:
        labels = ast.literal_eval(json_like).values()
        counts = Counter(labels)
        top_freq = max(counts.values())
        best = min(
            (lbl for lbl, n in counts.items() if n == top_freq),
            key=lambda lbl: self.PRIORITY.index(lbl)
        )
        return best

    def load_fin_data(self, path):
        df = pd.read_csv(path)
        df['sentiment'] = df['Decisions'].apply(self.single_sentiment)
        df["sentiment"] = pd.Categorical(
            df["sentiment"],
            categories=["negative", "positive", "neutral"],
            ordered=True
        )
        df["title_sentiment_class"] = self.sentiment_to_numeric(df["sentiment"])
        return df


    def create_sample_data(self):
        temp = self.load_fin_data(self.path)
        sample_data = {
            'title': temp['Title'],
            'original_sentiment': temp['title_sentiment_class'],
        }
        return pd.DataFrame(sample_data)

    def create_detailed_comparison_table(self, df):
        from sklearn.metrics import accuracy_score, classification_report
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
            logger.info(f"Recall - Negative: {report['Negative']['recall']:.3f}")
            logger.info(f"Recall - Neutral: {report['Neutral']['recall']:.3f}")
            logger.info(f"Recall - Positive: {report['Positive']['recall']:.3f}")
            logger.info(f"F1 Score - Negative: {report['Negative']['f1-score']:.3f}")
            logger.info(f"F1 Score - Neutral: {report['Neutral']['f1-score']:.3f}")
            logger.info(f"F1 Score - Positive: {report['Positive']['f1-score']:.3f}")
            logger.info(f"Support - Negative: {report['Negative']['support']:.3f}")
            logger.info(f"Support - Neutral: {report['Neutral']['support']:.3f}")
            logger.info(f"Support - Positive: {report['Positive']['support']:.3f}")
            df_report = pd.DataFrame(report)
            df_report.to_csv(f"reports/figures/sentiment_analysis/{self.ticker}_{name}_comparison_report.csv")
