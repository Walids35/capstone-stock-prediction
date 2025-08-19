from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from stock_prediction.config import PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialNewsAnalyzer:
    def __init__(self, data_path=None, df=None, stock_symbol=None):
        """
        Initialize the analyzer with either a file path or DataFrame

        Parameters:
        data_path (str): Path to CSV file
        df (pd.DataFrame): DataFrame with the data
        """
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")

        # Define feature groups
        self.sentiment_features = ["total_news_count",
                                   "finbert_majority_vote",
                                   "finbert_count_negative",
                                   "finbert_count_neutral",
                                   "finbert_count_positive",
                                   "finbert"
                                   "roberta_majority_vote",
                                   "roberta_count_negative",
                                   "roberta_count_neutral",
                                   "roberta_count_positive",
                                   "deberta_majority_vote", 
                                   "deberta_count_negative",
                                   "deberta_count_neutral",
                                   "deberta_count_positive",
                                   "svm_majority_vote",
                                   "svm_count_negative",
                                   "svm_count_neutral",
                                   "svm_count_positive",
                                   "rf_majority_vote",
                                   "rf_count_negative",
                                   "rf_count_neutral",
                                   "rf_count_positive",
                                   "lr_majority_vote",
                                   "lr_count_negative",
                                   "lr_count_neutral",
                                   "lr_count_positive",
                                   "finbert_label_positive_sum",
                                   "finbert_label_negative_sum",
                                   "finbert_label_neutral_sum",
                                   "roberta_label_positive_sum",
                                   "roberta_label_negative_sum",
                                   "roberta_label_neutral_sum",
                                   "deberta_label_negative_sum",
                                   "deberta_label_positive_sum",
                                   "deberta_label_neutral_sum"
        ]

        self.output_features = ['Float_Price', 'Binary_Price', 'Factor_Price', 'Delta_Price']

        self.stock_symbol = stock_symbol

        # Clean data
        self._clean_data()

    def _clean_data(self):
        """Clean and preprocess the data"""
        # Convert Date column to datetime if it exists
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())

        print(f"Dataset shape: {self.df.shape}")
        print(f"Available columns: {list(self.df.columns)}")

    def plot_output_distributions(self, figsize=(15, 10)):
        """Plot distributions of all output variables"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        for i, output in enumerate(self.output_features):
            if output in self.df.columns:
                ax = axes[i]

                if output == 'Binary_Price':
                    # Bar plot for binary target
                    counts = self.df[output].value_counts()
                    ax.bar(counts.index, counts.values)
                    ax.set_title(f'Distribution of {output}')
                else:
                    # Histogram and box plot for continuous variables
                    ax.hist(self.df[output].dropna(), bins=50, alpha=0.7, edgecolor='black')
                    ax.set_title(f'Distribution of {output}')
                    ax.set_xlabel(output)
                    ax.set_ylabel('Frequency')

                    # Add statistics
                    mean_val = self.df[output].mean()
                    median_val = self.df[output].median()
                    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                    ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
                    ax.legend()

        plt.tight_layout()
        plt.savefig(PROJ_ROOT / f"reports/figures/sentiment_vs_output/{self.stock_symbol}_output_distributions.png")

    def plot_sentiment_distributions(self, figsize=(20, 15)):
        """Plot distributions of sentiment features"""
        available_sentiment = [col for col in self.sentiment_features if col in self.df.columns]

        if not available_sentiment:
            print("No sentiment features found in the dataset")
            return

        n_cols = 4
        n_rows = (len(available_sentiment) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()

        for i, feature in enumerate(available_sentiment):
            if i < len(axes):
                ax = axes[i]
                data = self.df[feature].dropna()

                if data.dtype == 'object' or len(data.unique()) < 10:
                    # Categorical data
                    counts = data.value_counts()
                    ax.bar(range(len(counts)), counts.values)
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45)
                else:
                    # Continuous data
                    ax.hist(data, bins=30, alpha=0.7, edgecolor='black')

                ax.set_title(feature, fontsize=10)
                ax.tick_params(axis='x', labelsize=8)

        # Hide unused subplots
        for i in range(len(available_sentiment), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(PROJ_ROOT / f"reports/figures/sentiment_vs_output/{self.stock_symbol}_sentiment_distributions.png")

    def plot_correlation_matrix(self, figsize=(15, 12)):
        """Plot correlation matrix between sentiment features and outputs"""
        # Select numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        sentiment_cols = [col for col in self.sentiment_features if col in numeric_cols]
        output_cols = [col for col in self.output_features if col in numeric_cols]

        all_cols = sentiment_cols + output_cols
        correlation_data = self.df[all_cols].corr()

        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Correlation Matrix: Sentiment Features vs Output Variables')
        plt.tight_layout()
        
        plt.savefig(PROJ_ROOT / f"reports/figures/sentiment_vs_output/{self.stock_symbol}_correlation_matrix.png")

        # Print strongest correlations with outputs
        print("\nStrongest correlations with output variables:")
        for output in output_cols:
            if output in correlation_data.columns:
                corr_with_output = correlation_data[output].drop(output).abs().sort_values(ascending=False)
                print(f"\n{output}:")
                print(corr_with_output.head(5))

    def plot_sentiment_vs_outputs(self, figsize=(20, 15)):
        """Create scatter plots of sentiment features vs output variables"""
        # Select numeric sentiment features
        numeric_cols = self.df.columns
        sentiment_cols = [col for col in self.sentiment_features if col in numeric_cols]
        output_cols = [col for col in self.output_features if col in numeric_cols and col != 'Binary_Price']

        if not sentiment_cols or not output_cols:
            print("No suitable numeric features found for scatter plots")
            return

        fig, axes = plt.subplots(len(output_cols), len(sentiment_cols), figsize=figsize)
        if len(output_cols) == 1:
            axes = axes.reshape(1, -1)

        for i, output in enumerate(output_cols):
            for j, sentiment in enumerate(sentiment_cols):
                ax = axes[i, j] if len(output_cols) > 1 else axes[j]

                # Create scatter plot
                ax.scatter(self.df[sentiment], self.df[output], alpha=0.5, s=20)

                # Add trend line
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        self.df[sentiment].dropna(), self.df[output].dropna()
                    )
                    line = slope * self.df[sentiment] + intercept
                    ax.plot(self.df[sentiment], line, 'r', alpha=0.8)
                    ax.set_title(f'{sentiment} vs {output}\nR²={r_value**2:.3f}', fontsize=10)
                except:
                    ax.set_title(f'{sentiment} vs {output}', fontsize=10)

                ax.set_xlabel(sentiment, fontsize=8)
                ax.set_ylabel(output, fontsize=8)
                ax.tick_params(axis='both', labelsize=6)

        plt.tight_layout()
        

    def plot_binary_target_analysis(self, figsize=(15, 10)):
        """Analyze sentiment features by binary target groups"""
        if 'Binary_Price' not in self.df.columns:
            print("Binary_Price column not found")
            return

        # Select numeric sentiment features
        numeric_cols = self.df.columns
        sentiment_cols = [col for col in self.sentiment_features if col in numeric_cols]
        cols = ["finbert_label_positive_sum",
                                   "finbert_label_negative_sum",
                                   "finbert_label_neutral_sum",
                                   "roberta_label_positive_sum",
                                   "roberta_label_negative_sum",
                                   "roberta_label_neutral_sum",
                                   "deberta_label_negative_sum",
                                   "deberta_label_positive_sum",
                                   "deberta_label_neutral_sum"]
        sentiment_cols = [col for col in sentiment_cols if col in cols]

        if not sentiment_cols:
            print("No numeric sentiment features found")
            return

        n_cols = 4
        n_rows = (len(sentiment_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.ravel()

        for i, feature in enumerate(sentiment_cols):
            if i < len(axes):
                ax = axes[i]

                # Box plot by binary target
                data_0 = self.df[self.df['Binary_Price'] == 0][feature].dropna()
                data_1 = self.df[self.df['Binary_Price'] == 1][feature].dropna()

                ax.boxplot([data_0, data_1], labels=['0', '1'])
                ax.set_title(f'{feature} by Binary_Price', fontsize=10)
                ax.set_xlabel('Binary_Price', fontsize=8)
                ax.set_ylabel(feature, fontsize=8)

        # Hide unused subplots
        for i in range(len(sentiment_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        
        plt.savefig(PROJ_ROOT / f"reports/figures/sentiment_vs_output/{self.stock_symbol}_binary_target_analysis.png")

    def plot_time_series_analysis(self, figsize=(15, 8)):
        """Plot time series of outputs and sentiment if Date column exists"""
        if 'Date' not in self.df.columns:
            print("Date column not found for time series analysis")
            return

        # Sort by date
        df_sorted = self.df.sort_values('Date')

        # Plot outputs over time
        output_cols = [col for col in self.output_features if col in self.df.columns and col != 'TargetBinary']

        if output_cols:
            fig, axes = plt.subplots(len(output_cols), 1, figsize=figsize, sharex=True)
            if len(output_cols) == 1:
                axes = [axes]

            for i, output in enumerate(output_cols):
                axes[i].plot(df_sorted['Date'], df_sorted[output], alpha=0.7)
                axes[i].set_title(f'{output} over Time')
                axes[i].set_ylabel(output)
                axes[i].grid(True, alpha=0.3)

            axes[-1].set_xlabel('Date')
            plt.tight_layout()
            

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("=" * 60)
        print("FINANCIAL NEWS SENTIMENT ANALYSIS REPORT")
        print("=" * 60)

        print(f"\nDataset Overview:")
        print(f"- Total samples: {len(self.df)}")
        print(f"- Total features: {len(self.df.columns)}")
        print(f"- Date range: {self.df['Date'].min()} to {self.df['Date'].max()}" if 'Date' in self.df.columns else "- No date information")

        # Output variables summary
        print(f"\nOutput Variables Summary:")
        for output in self.output_features:
            if output in self.df.columns:
                if output == 'TargetBinary':
                    counts = self.df[output].value_counts()
                    print(f"- {output}: {counts.to_dict()}")
                else:
                    print(f"- {output}: Mean={self.df[output].mean():.4f}, Std={self.df[output].std():.4f}")

        # Sentiment features summary
        available_sentiment = [col for col in self.sentiment_features if col in self.df.columns]
        print(f"\nSentiment Features Available: {len(available_sentiment)}")
        print(f"- {', '.join(available_sentiment[:5])}{'...' if len(available_sentiment) > 5 else ''}")

        print("\n" + "=" * 60)


    def run_complete_analysis(self):
        """Run all analysis functions"""
        print("Starting comprehensive financial news sentiment analysis...")

        # Generate summary
        self.generate_summary_report()

        # Run all plotting functions
        print("\n1. Plotting output distributions...")
        self.plot_output_distributions()

        print("\n2. Plotting sentiment feature distributions...")
        self.plot_sentiment_distributions()

        print("\n3. Creating correlation matrix...")
        self.plot_correlation_matrix()

        print("\n4. Creating sentiment vs output scatter plots...")
        self.plot_sentiment_vs_outputs()

        print("\n5. Analyzing binary target groups...")
        self.plot_binary_target_analysis()

        print("\nAnalysis complete!")

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "TSLA_preprocessed_dataset_with_features.csv"
):
    logger.info("Generating plot from data...")
    df = pd.read_csv(input_path)
    analyzer = FinancialNewsAnalyzer(df=df, stock_symbol="TSLA")
    analyzer.run_complete_analysis()
    logger.success("Plot generation complete.") 


if __name__ == "__main__":
    app()
