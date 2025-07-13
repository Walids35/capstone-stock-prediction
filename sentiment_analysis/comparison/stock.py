import pandas as pd
import re
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from venny4py.venny4py import *
from models.base import BaseSentimentComparison
from loguru import logger

class StockSentimentComparison(BaseSentimentComparison):
    def __init__(self, path, source_name):
        super().__init__(path, source_name)

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