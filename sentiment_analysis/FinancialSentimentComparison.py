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

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialSentimentComparison:
    def __init__(self, path, ticker):
        """Initialize the sentiment analysis models"""
        logger.info("Loading sentiment analysis models...")
        self.path = path
        self.ticker = ticker
        self.PRIORITY = ("positive", "negative", "neutral")
        # Initialize models
        self.models = {}

        # FinBERT - specialized for financial sentiment
        self.models['finbert'] = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )

        # RoBERTa - general sentiment model
        self.models['roberta'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        # DeBERTa - advanced model
        self.models['deberta'] = pipeline(
            "sentiment-analysis",
            model="nickmuchi/deberta-v3-base-finetuned-finance-text-classification"
        )

        logger.success("All models loaded successfully!")

    def sentiment_to_numeric(self, series) -> pd.Series:
        """Map 'negative' → -1, 'neutral' → 0, 'positive' → 1 (Int8)."""
        mapping = {"negative": 0,"positive": 2,"neutral":1}

        if pd.api.types.is_categorical_dtype(series):
            return series.astype(str).map(mapping).astype("Int8")
        else:
            return series.map(mapping).astype("Int8")

    def single_sentiment(self, json_like: str) -> str:
        """
        Convert a string such as
            '{"banking": "positive", "Centrum Broking": "neutral"}'
        to one of: 'positive' · 'negative' · 'neutral'.

        Rule:
        1. Take the majority label among all entity-level labels.
        2. If there is a tie, pick whichever comes first in PRIORITY.
        """
        labels   = ast.literal_eval(json_like).values()   # dict → iterable of labels
        counts   = Counter(labels)
        top_freq = max(counts.values())

        # tie-aware selection
        best = min(
            (lbl for lbl, n in counts.items() if n == top_freq),
            key=lambda lbl: self.PRIORITY.index(lbl)
        )
        return best
    
    def load_fin_data(self, path):
        df=pd.read_csv(path)
        df['sentiment'] = df['Decisions'].apply(self.single_sentiment)
        df["sentiment"] = pd.Categorical(
                    df["sentiment"],
                    categories=["negative","positive","neutral"],
                    ordered=True
        )
        df["title_sentiment_class"] = self.sentiment_to_numeric(df["sentiment"])
        return df

    def create_sample_data(self):
        """Create sample financial news dataset"""
        temp = self.load_fin_data(self.path)
        sample_data = {
            'title': temp['Title'],
            'original_sentiment': temp['title_sentiment_class'],
        }

        return pd.DataFrame(sample_data)

    def map_sentiment_scores(self, model_name, predictions):
        """Map model predictions to 0, 1, 2 scale"""
        mapped_scores = []

        for pred in predictions:
            if model_name == 'finbert':
                # FinBERT labels: positive, negative, neutral
                if pred['label'] == 'positive':
                    mapped_scores.append(2)
                elif pred['label'] == 'negative':
                    mapped_scores.append(0)
                else:
                    mapped_scores.append(1)

            elif model_name == 'roberta':
                # RoBERTa labels: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
                if pred['label'] == 'positive':
                    mapped_scores.append(2)
                elif pred['label'] == 'negative':
                    mapped_scores.append(0)
                else:
                    mapped_scores.append(1)

            else:  # deberta
                # DeBERTa typically uses POSITIVE, NEGATIVE
                if pred['label'].upper() in ['POSITIVE', 'POS','BULLISH']:
                    mapped_scores.append(2)
                elif pred['label'].upper() in ['NEGATIVE', 'NEG','BEARISH']:
                    mapped_scores.append(0)
                else:
                    mapped_scores.append(1)

        return mapped_scores

    def analyze_sentiments(self, df):
        """Run all models on the dataset"""
        logger.info("Running sentiment analysis with all models...")

        texts = df['title'].tolist()

        # Run each model
        for model_name, model in self.models.items():
            logger.info(f"Processing with {model_name}...")

            predictions = model(texts)
            mapped_scores = self.map_sentiment_scores(model_name, predictions)

            # Add to dataframe
            df[f'{model_name}_sentiment'] = mapped_scores
            df[f'{model_name}_confidence'] = [pred['score'] for pred in predictions]

        return df

    def calculate_agreement_matrix(self, df):
        """Calculate agreement between models"""
        models = ['original_sentiment', 'finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']
        agreement_matrix = pd.DataFrame(index=models, columns=models)

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i <= j:
                    agreement = accuracy_score(df[model1], df[model2])
                    agreement_matrix.loc[model1, model2] = agreement
                    agreement_matrix.loc[model2, model1] = agreement

        return agreement_matrix.astype(float)

    def create_confusion_matrices(self, df):
        """Create confusion matrices for model comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confusion Matrices: Models vs Original Labels', fontsize=16, fontweight='bold')

        models = ['finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']
        model_names = ['FinBERT', 'RoBERTa', 'DeBERTa']

        for i, (model, name) in enumerate(zip(models, model_names)):
            row = i // 2
            col = i % 2

            cm = confusion_matrix(df['original_sentiment'], df[model])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Neutral', 'Positive'],
                       yticklabels=['Negative', 'Neutral', 'Positive'],
                       ax=axes[row, col])
            axes[row, col].set_title(f'{name} vs Original')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')

        # Hide the last subplot
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"reports/figures/sentiment_analysis/SentFiN_confusion_matrix_{self.ticker}.png")

    def create_agreement_heatmap(self, agreement_matrix):
        """Create heatmap of model agreements"""
        plt.figure(figsize=(10, 8))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)

        sns.heatmap(agreement_matrix, mask=mask, annot=True, fmt='.3f',
                   cmap='RdYlBu_r', center=0.5, square=True,
                   xticklabels=['Original', 'FinBERT', 'RoBERTa', 'DeBERTa'],
                   yticklabels=['Original', 'FinBERT', 'RoBERTa', 'DeBERTa'])

        plt.title('Model Agreement Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        plt.savefig(f"reports/figures/sentiment_analysis/SentFiN_agreement_heatmap_{self.ticker}.png")

    def create_venn_diagrams(self, df):
        """Create and save separate Venn diagrams for each sentiment class using venny4py."""
        sentiments = [
            (0, 'Negative'),
            (1, 'Neutral'),
            (2, 'Positive')
        ]

        for sent_val, sentiment in sentiments:
            original_set = set(df[df['original_sentiment'] == sent_val].index)
            finbert_set = set(df[df['finbert_sentiment'] == sent_val].index)
            roberta_set = set(df[df['roberta_sentiment'] == sent_val].index)
            deberta_set = set(df[df['deberta_sentiment'] == sent_val].index)

            sets = {
                'Original': original_set,
                'Finbert': finbert_set,
                'Roberta': roberta_set,
                'Deberta': deberta_set
            }

            venny4py(sets=sets)
            plt.title(f'{sentiment} Predictions', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"reports/figures/sentiment_analysis/SentFiN_venn_diagram_{sentiment}_{self.ticker}.png")
            plt.clf()  # Clear the current figure for the next plot

    def create_confidence_analysis(self, df):
        """Analyze model confidence scores"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Confidence distribution by model
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

        # Average confidence by agreement
        axes[1, 1].clear()

        # Calculate agreement with original
        for model in models:
            df[f'{model}_agrees'] = (df['original_sentiment'] == df[f'{model}_sentiment']).astype(int)

            agree_conf = df[df[f'{model}_agrees'] == 1][f'{model}_confidence'].mean()
            disagree_conf = df[df[f'{model}_agrees'] == 0][f'{model}_confidence'].mean()

            axes[1, 1].bar([f'{model.title()}\nAgree', f'{model.title()}\nDisagree'],
                          [agree_conf, disagree_conf], alpha=0.7)

        axes[1, 1].set_title('Average Confidence: Agreement vs Disagreement')
        axes[1, 1].set_ylabel('Average Confidence')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        plt.savefig(f"reports/figures/sentiment_analysis/SentFiN_confidence_analysis_{self.ticker}.png")

    def create_detailed_comparison_table(self, df):
        """Create detailed comparison table"""
        logger.info("\n" + "="*80)
        logger.success("DETAILED MODEL COMPARISON REPORT")
        logger.info("="*80)

        models = ['finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']
        model_names = ['FinBERT', 'RoBERTa', 'DeBERTa']

        for model, name in zip(models, model_names):
            logger.info(f"\n{name} Performance:")
            logger.info("-" * 30)

            # Accuracy
            accuracy = accuracy_score(df['original_sentiment'], df[model])
            logger.success(f"Overall Accuracy: {accuracy:.3f}")

            # Classification report
            report = classification_report(df['original_sentiment'], df[model],
                                         target_names=['Negative', 'Neutral', 'Positive'],
                                         output_dict=True)

            logger.info(f"Precision - Negative: {report['Negative']['precision']:.3f}")
            logger.info(f"Precision - Neutral: {report['Neutral']['precision']:.3f}")
            logger.info(f"Precision - Positive: {report['Positive']['precision']:.3f}")

    def run_complete_analysis(self, df=None):
        """Run the complete analysis pipeline"""
        if df is None:
            logger.info("Creating sample financial news dataset...")
            df = self.create_sample_data()

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Original sentiment distribution:\n{df['original_sentiment'].value_counts().sort_index()}")

        # Run sentiment analysis
        df = self.analyze_sentiments(df)

        # Calculate agreements
        agreement_matrix = self.calculate_agreement_matrix(df)

        # Create visualizations
        logger.info("\nCreating visualizations...")

        # 1. Confusion matrices
        self.create_confusion_matrices(df)

        # 2. Agreement heatmap
        self.create_agreement_heatmap(agreement_matrix)

        # 3. Venn diagrams
        try:
            self.create_venn_diagrams(df)
        except ImportError:
            logger.info("Note: Install matplotlib-venn for Venn diagrams: pip install matplotlib-venn")

        # 4. Confidence analysis
        self.create_confidence_analysis(df)

        # 5. Detailed comparison
        self.create_detailed_comparison_table(df)

        return df, agreement_matrix