from pathlib import Path
from typing import List, Optional, Dict, Any
import sys

# Add the project root to the path to ensure imports work
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from datetime import datetime, date

from stock_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def load_price_data(stock_symbol: str) -> pd.DataFrame:
    """Load price data for a given stock symbol."""
    price_file = RAW_DATA_DIR / "price data" / f"{stock_symbol}_price_data.csv"
    
    if not price_file.exists():
        raise FileNotFoundError(f"Price data file not found: {price_file}")
    
    logger.info(f"Loading price data from {price_file}")
    df = pd.read_csv(price_file)
    
    # Ensure date column is properly formatted
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def load_sentiment_data(stock_symbol: str) -> pd.DataFrame:
    """Load sentiment analysis results for a given stock symbol."""
    # Try different possible file names
    possible_files = [
        INTERIM_DATA_DIR / "models" / f"{stock_symbol}_financial_news_sentiment_analysis_results_with_classical_models.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            logger.info(f"Loading sentiment data from {file_path}")
            return pd.read_csv(file_path)
    
    raise FileNotFoundError(f"Sentiment data file not found for {stock_symbol}")


def extract_date_column(news_df: pd.DataFrame) -> pd.Series:
    """Extract and standardize date column from news dataframe."""
    date_columns = ['date', 'datetime', 'Date', 'DateTime']
    
    for col in date_columns:
        if col in news_df.columns:
            return pd.to_datetime(news_df[col]).dt.date
    
    # Fallback: try to find any datetime-like column
    for col in news_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                return pd.to_datetime(news_df[col]).dt.date
            except:
                continue
    
    # Last resort: use the 4th column (index 3) as mentioned in original code
    if len(news_df.columns) > 3:
        try:
            return pd.to_datetime(news_df.iloc[:, 3]).dt.date
        except:
            pass
    
    raise ValueError("Could not find a valid date column in the news dataframe")


def identify_model_columns(news_df: pd.DataFrame) -> Dict[str, str]:
    """Identify sentiment model columns in the dataframe."""
    model_names = ["deberta", "finbert", "roberta", "svm", "lr", "rf"]
    model_cols = {}
    
    for name in model_names:
        matches = [col for col in news_df.columns if name.lower() in col.lower()]
        if matches:
            model_cols[name] = matches[0]
    
    # If we didn't find expected models, try to use the last 6 columns
    if len(model_cols) < 3:
        logger.warning(f"Only found {len(model_cols)} model columns, using last 6 columns as fallback")
        last_cols = list(news_df.columns[-6:])
        for i, name in enumerate(model_names[:len(last_cols)]):
            model_cols[name] = last_cols[i]
    
    return model_cols


def aggregate_sentiment_by_date(
    news_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate sentiment scores by date, then set 'mean_score' to the max
       of all per-model mean scores (i.e. take the highest confidence)."""
    news_df = news_df.copy()
    news_df["date_only"] = extract_date_column(news_df)

    # Base: total news count per date
    result_df = (
        news_df
        .groupby("date_only")
        .size()
        .reset_index(name="total_news_count")
    )

    def majority_vote(series):
        return series.mode().iloc[0] if not series.mode().empty else None

    # Group by date and apply aggregations
    aggregation = {
        # Combined aggregations for each sentiment model (majority vote, min, max, and count by label)
        'finbert_sentiment': [
            majority_vote, 'min', 'max',
            lambda x: (x == 2).sum(), 
            lambda x: (x == 0).sum(), 
            lambda x: (x == 1).sum()
        ],
        'roberta_sentiment': [
            majority_vote, 'min', 'max',
            lambda x: (x == 2).sum(), 
            lambda x: (x == 0).sum(), 
            lambda x: (x == 1).sum()
        ],
        'deberta_sentiment': [
            majority_vote, 'min', 'max',
            lambda x: (x == 2).sum(), 
            lambda x: (x == 0).sum(), 
            lambda x: (x == 1).sum()
        ],
        'svm_sentiment': [
            majority_vote, 'min', 'max',
            lambda x: (x == 2).sum(), 
            lambda x: (x == 0).sum(), 
            lambda x: (x == 1).sum()
        ],
        'rf_sentiment': [
            majority_vote, 'min', 'max',
            lambda x: (x == 2).sum(), 
            lambda x: (x == 0).sum(), 
            lambda x: (x == 1).sum()
        ],
        'lr_sentiment': [
            majority_vote, 'min', 'max',
            lambda x: (x == 2).sum(), 
            lambda x: (x == 0).sum(), 
            lambda x: (x == 1).sum()
        ],

        # Confidence sum
        'finbert_label_positive': 'sum',
        'finbert_label_negative': 'sum',
        'finbert_label_neutral': 'sum',
        'roberta_label_positive': 'sum',
        'roberta_label_negative': 'sum',
        'roberta_label_neutral': 'sum',
        'deberta_label_bearish': 'sum',
        'deberta_label_bullish': 'sum',
        'deberta_label_neutral': 'sum',
    }

    # Perform aggregation
    grouped = news_df.groupby('date_only').agg(aggregation)
    
    # Flatten column names first
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    # Now rename the flattened columns
    grouped.rename(columns={
        # Majority vote, min, max columns
        "finbert_sentiment_majority_vote": "finbert_majority_vote",
        "finbert_sentiment_min": "finbert_min",
        "finbert_sentiment_max": "finbert_max",
        "roberta_sentiment_majority_vote": "roberta_majority_vote",
        "roberta_sentiment_min": "roberta_min",
        "roberta_sentiment_max": "roberta_max",
        "deberta_sentiment_majority_vote": "deberta_majority_vote",
        "deberta_sentiment_min": "deberta_min",
        "deberta_sentiment_max": "deberta_max",
        "svm_sentiment_majority_vote": "svm_majority_vote",
        "svm_sentiment_min": "svm_min",
        "svm_sentiment_max": "svm_max",
        "lr_sentiment_majority_vote": "lr_majority_vote",
        "lr_sentiment_min": "lr_min",
        "lr_sentiment_max": "lr_max",
        "rf_sentiment_majority_vote": "rf_majority_vote",
        "rf_sentiment_min": "rf_min",
        "rf_sentiment_max": "rf_max",
        
        # Count columns
        "finbert_sentiment_<lambda_0>": "finbert_count_negative",
        "finbert_sentiment_<lambda_1>": "finbert_count_neutral",
        "finbert_sentiment_<lambda_2>": "finbert_count_positive",
        "deberta_sentiment_<lambda_0>": "deberta_count_negative",
        "deberta_sentiment_<lambda_1>": "deberta_count_neutral",
        "deberta_sentiment_<lambda_2>": "deberta_count_positive",
        "roberta_sentiment_<lambda_0>": "roberta_count_negative",
        "roberta_sentiment_<lambda_1>": "roberta_count_neutral",
        "roberta_sentiment_<lambda_2>": "roberta_count_positive",
        "svm_sentiment_<lambda_0>": "svm_count_negative",
        "svm_sentiment_<lambda_1>": "svm_count_neutral",
        "svm_sentiment_<lambda_2>": "svm_count_positive",
        "lr_sentiment_<lambda_0>": "lr_count_negative",
        "lr_sentiment_<lambda_1>": "lr_count_neutral",
        "lr_sentiment_<lambda_2>": "lr_count_positive",
        "rf_sentiment_<lambda_0>": "rf_count_negative",
        "rf_sentiment_<lambda_1>": "rf_count_neutral",
        "rf_sentiment_<lambda_2>": "rf_count_positive",
        "deberta_label_bearish_sum": "deberta_label_negative_sum",
        "deberta_label_bullish_sum": "deberta_label_positive_sum",
        "deberta_label_neutral_sum": "deberta_label_neutral_sum",
    }, inplace=True)
    
    grouped = grouped.reset_index()

    result_df = result_df.merge(grouped, on="date_only", how="left")
    
    if len(result_df.columns) == 1:
        raise ValueError("No valid model columns found for aggregation")

    return result_df


def merge_price_and_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price data with aggregated sentiment data in horizontal format."""
    # Ensure we have a date column in price data
    date_col = None
    for col in ['Date', 'date']:
        if col in price_df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("No date column found in price data")
    
    # Convert price date to date only for merging
    price_df = price_df.copy()
    price_df['date_only'] = pd.to_datetime(price_df[date_col]).dt.date
    
    # Merge on date - this will now create one row per date with all model metrics as columns
    merged_df = price_df.merge(sentiment_df, on='date_only', how='left')
    
    # Sort by date for better readability
    merged_df = merged_df.sort_values('date_only')
    
    return merged_df


@app.command()
def main(
    stock_symbol: str = typer.Option("TSLA", help="Stock symbol to process"),
    input_path: Optional[Path] = typer.Option(None, help="Custom path to price data file"),
    news_path: Optional[Path] = typer.Option(None, help="Custom path to sentiment data file"),
    output_path: Optional[Path] = typer.Option(None, help="Custom output path"),
    verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """
    Process stock price data and sentiment analysis results to create a preprocessed dataset.
    
    This script combines price data with sentiment analysis results from multiple models,
    aggregating sentiment scores by date and merging with price information.
    """
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    
    logger.info(f"Processing dataset for {stock_symbol}...")
    
    try:
        # Load data
        if input_path:
            logger.info(f"Loading price data from custom path: {input_path}")
            price_df = pd.read_csv(input_path)
        else:
            price_df = load_price_data(stock_symbol)
        
        if news_path:
            logger.info(f"Loading sentiment data from custom path: {news_path}")
            news_df = pd.read_csv(news_path)
        else:
            news_df = load_sentiment_data(stock_symbol)
        
        # Identify model columns
        model_cols = identify_model_columns(news_df)
        logger.info(f"Found model columns: {list(model_cols.keys())}")
        
        # Aggregate sentiment by date
        sentiment_agg_df = aggregate_sentiment_by_date(news_df)
        logger.info(f"Aggregated sentiment data shape: {sentiment_agg_df.shape}")
        
        # Merge with price data
        merged_df = merge_price_and_sentiment(price_df, sentiment_agg_df)
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        # Determine output path
        if output_path:
            final_output_path = output_path
        else:
            final_output_path = PROCESSED_DATA_DIR / f"{stock_symbol}_preprocessed_dataset.csv"
        
        # Ensure output directory exists
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the processed dataset
        merged_df.to_csv(final_output_path, index=False)
        logger.success(f"Processed dataset saved to: {final_output_path}")
        
        # Print summary statistics
        logger.info(f"Dataset summary:")
        logger.info(f"  - Total rows: {len(merged_df)}")
        logger.info(f"  - Date range: {merged_df['date_only'].min()} to {merged_df['date_only'].max()}")
        logger.info(f"  - Models included: {list(model_cols.keys())}")
        logger.info(f"  - Total columns: {len(merged_df.columns)}")
        
        # Show column structure
        sentiment_cols = [col for col in merged_df.columns if any(model in col for model in model_cols.keys())]
        logger.info(f"  - Sentiment columns created: {len(sentiment_cols)}")
        
        # Show sample of the data
        logger.info("Sample of processed data (horizontal format):")
        print(merged_df.head())
        
        # Show column names for verification
        logger.info("Column structure:")
        for col in merged_df.columns:
            logger.info(f"    - {col}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise
    

if __name__ == "__main__":
    app()
