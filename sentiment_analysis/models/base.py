import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.bert_model import BertModel
from abc import ABC, abstractmethod
from matplotlib_venn import venn3
from loguru import logger

class BaseSentimentComparison(ABC):
    def __init__(self, path, ticker=None):
        """Initialize the sentiment analysis models"""
        self.path = path
        self.ticker = ticker
        self.models = {}
        self._load_models()

    def _load_models(self):
        # FinBERT
        finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.models['finbert'] = BertModel(finbert_model, finbert_tokenizer)
        # RoBERTa
        roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.models['roberta'] = BertModel(roberta_model, roberta_tokenizer)
        # DeBERTa
        deberta_model = AutoModelForSequenceClassification.from_pretrained("nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
        deberta_tokenizer = AutoTokenizer.from_pretrained("nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
        self.models['deberta'] = BertModel(deberta_model, deberta_tokenizer)

    @abstractmethod
    def load_fin_data(self, path):
        pass

    @abstractmethod
    def create_sample_data(self):
        pass

    @abstractmethod
    def sentiment_to_numeric(self, series) -> pd.Series:
        pass

    def map_sentiment_scores(self, model_name, predictions):
        mapped_scores = []
        for pred in predictions:
            if model_name == 'finbert':
                if pred['label'].lower() == 'positive':
                    mapped_scores.append(2)
                elif pred['label'].lower() == 'negative':
                    mapped_scores.append(0)
                else:
                    mapped_scores.append(1)
            elif model_name == 'roberta':
                if pred['label'].lower() == 'positive':
                    mapped_scores.append(2)
                elif pred['label'].lower() == 'negative':
                    mapped_scores.append(0)
                else:
                    mapped_scores.append(1)
            else:  # deberta
                if pred['label'].upper() in ['POSITIVE', 'POS', 'BULLISH']:
                    mapped_scores.append(2)
                elif pred['label'].upper() in ['NEGATIVE', 'NEG', 'BEARISH']:
                    mapped_scores.append(0)
                else:
                    mapped_scores.append(1)
        return mapped_scores

    def analyze_sentiments(self, df):
        texts = df['title'].tolist()
        for model_name, model in self.models.items():
            logger.info(f"Analyzing {model_name} sentiments...")
            predictions = []
            for text in texts:
                logger.info(text)
                prediction = model.generate(text)
                predictions.append(prediction)
            mapped_scores = self.map_sentiment_scores(model_name, predictions)
            df[f'{model_name}_sentiment'] = mapped_scores
            df[f'{model_name}_confidence'] = [max(pred['probabilities'].values()) for pred in predictions]

            for key in predictions[0]['probabilities'].keys():
                df[f'{model_name}_label_{key}'] = [pred['probabilities'][key] for pred in predictions]
        return df

    def calculate_agreement_matrix(self, df):
        models = ['finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']
        agreement_matrix = pd.DataFrame(index=models, columns=models)
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i <= j:
                    agreement = accuracy_score(df[model1], df[model2])
                    agreement_matrix.loc[model1, model2] = agreement
                    agreement_matrix.loc[model2, model1] = agreement
        return agreement_matrix.astype(float)

    def create_confusion_matrices(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Confusion Matrices: Pairwise Model Comparisons', fontsize=16, fontweight='bold')
        
        # Define all pairwise comparisons
        comparisons = [
            ('finbert_sentiment', 'roberta_sentiment', 'FinBERT vs RoBERTa'),
            ('finbert_sentiment', 'deberta_sentiment', 'FinBERT vs DeBERTa'),
            ('roberta_sentiment', 'deberta_sentiment', 'RoBERTa vs DeBERTa')
        ]
        
        for i, (model1, model2, title) in enumerate(comparisons):
            cm = confusion_matrix(df[model1], df[model2])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Neutral', 'Positive'],
                       yticklabels=['Negative', 'Neutral', 'Positive'],
                       ax=axes[i])
            axes[i].set_title(title)
            axes[i].set_xlabel(f'{model2.replace("_sentiment", "").title()} Predicted')
            axes[i].set_ylabel(f'{model1.replace("_sentiment", "").title()} Actual')
        
        plt.tight_layout()
        plt.savefig(f"reports/figures/sentiment_analysis/{self.ticker}_confusion_matrices.png")

    def create_agreement_heatmap(self, agreement_matrix):
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)
        sns.heatmap(agreement_matrix, mask=mask, annot=True, fmt='.3f',
                   cmap='RdYlBu_r', center=0.5, square=True,
                   xticklabels=['FinBERT', 'RoBERTa', 'DeBERTa'],
                   yticklabels=['FinBERT', 'RoBERTa', 'DeBERTa'])
        plt.title('Model Agreement Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"reports/figures/sentiment_analysis/{self.ticker}_agreement_heatmap.png")

    def create_venn_diagrams(self, df, prefix="venn_diagram"):
        from matplotlib_venn import venn3
        
        sentiments = [
            (0, 'Negative'),
            (1, 'Neutral'),
            (2, 'Positive')
        ]
        for sent_val, sentiment in sentiments:
            finbert_set = set(df[df['finbert_sentiment'] == sent_val].index)
            roberta_set = set(df[df['roberta_sentiment'] == sent_val].index)
            deberta_set = set(df[df['deberta_sentiment'] == sent_val].index)
            
            plt.figure(figsize=(8, 8))
            venn3([finbert_set, roberta_set, deberta_set], 
                  set_labels=('FinBERT', 'RoBERTa', 'DeBERTa'))
            plt.title(f'{sentiment} Predictions', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"reports/figures/sentiment_analysis/{self.ticker}_{prefix}_{sentiment}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def create_confidence_analysis(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        models = ['finbert', 'roberta', 'deberta']
        model_names = ['FinBERT', 'RoBERTa', 'DeBERTa']
        for i, (model, name) in enumerate(zip(models, model_names)):
            row = i // 2
            col = i % 2
            confidence_col = f'{model}_confidence'
            sentiment_col = f'{model}_sentiment'
            for sentiment in [0, 1, 2]:
                sentiment_data = df[df[sentiment_col] == sentiment][confidence_col]
                axes[row, col].hist(sentiment_data, alpha=0.7,
                                   label=f'Sentiment {sentiment}', bins=15)
            axes[row, col].set_title(f'{name} Confidence Distribution')
            axes[row, col].set_xlabel('Confidence Score')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        # Show overall confidence distribution for all models
        axes[1, 1].clear()
        for model, name in zip(models, model_names):
            confidence_col = f'{model}_confidence'
            axes[1, 1].hist(df[confidence_col], alpha=0.6, label=name, bins=20)
        axes[1, 1].set_title('Overall Confidence Distribution by Model')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"reports/figures/sentiment_analysis/{self.ticker}_confidence_analysis.png")

    @abstractmethod
    def create_detailed_comparison_table(self, df):
        pass

    def run_complete_analysis(self, df=None, venn_prefix="venn_diagram"):
        if df is None:
            df = self.create_sample_data()
        logger.info(f"Dataset shape: {df.shape}")
        # Analyze sentiments across all three models
        df = self.analyze_sentiments(df)
        agreement_matrix = self.calculate_agreement_matrix(df)
        logger.info("\nCreating visualizations...")
        self.create_confusion_matrices(df)
        self.create_agreement_heatmap(agreement_matrix)
        self.create_venn_diagrams(df, prefix=venn_prefix)
        self.create_confidence_analysis(df)
        self.create_detailed_comparison_table(df)
        return df, agreement_matrix 