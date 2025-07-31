from pathlib import Path
from loguru import logger
from stock_prediction.config import *
import pandas as pd
from models.base import BaseSentimentComparison
from models.bert_model import BertModel
from comparison.financial import FinancialSentimentComparison
from comparison.stock import StockSentimentComparison
from trainers.model_trainer import ModelTrainer
import warnings

warnings.filterwarnings("ignore")

def main():
    
    # Run the sentiment analysis of Roberta, FinBERT and DeBERTa models with SentFiN-v1.1 dataset and save the results
    dataset_paths = [
        PROJ_ROOT / "data/raw/human_labeled/SEntFiN-v1.1.csv",
    ]

    for csv_path in dataset_paths:
        logger.info(f"\n=== Processing: {csv_path} ===")
        source_name = Path(csv_path).stem

        analyzer = FinancialSentimentComparison(csv_path, "SentFiN-v1.1")
        results_df, agreement_matrix = analyzer.run_complete_analysis()

        results_file   = f"{source_name}_sentiment_analysis_results.csv"
        agree_file     = f"{source_name}_model_agreement_matrix.csv"

        results_df.to_csv(f"{PROJ_ROOT}/reports/output/{results_file}", index=False)
        agreement_matrix.to_csv(f"{PROJ_ROOT}/reports/output/{agree_file}")

        logger.success("→  Analysis complete!  Files saved:")
        logger.success(f"   – {results_file}")
        logger.success(f"   – {agree_file}")
        logger.success(f"     DataFrame shape: {results_df.shape}")
        logger.success("     Columns:")
        for col in results_df.columns:
            logger.success(f"       · {col}")

    logger.success("\n🎉  All datasets processed!")
    
    # Train the models with the results of the sentiment analysis
    results_df = pd.read_csv(f"{PROJ_ROOT}/reports/output/SEntFiN-v1.1_sentiment_analysis_results.csv")
    x_cols = ["finbert_sentiment","finbert_confidence", "roberta_sentiment", "roberta_confidence", "deberta_sentiment", "deberta_confidence"]
    target_col = "original_sentiment"
    model_trainer = ModelTrainer(results_df, x_cols, target_col, use_grid_search=True)

    model_trainer.train_logistic_regression()
    model_trainer.train_svm()
    model_trainer.train_random_forest()

    # Running Bert-based models for the financial news of each ticker
    from pathlib import Path
    dataset_paths = [
        PROJ_ROOT / "data/raw/news data/AAPL_financial_news.csv",
        PROJ_ROOT / "data/raw/news data/TSLA_financial_news.csv",
        PROJ_ROOT / "data/raw/news data/AMZN_financial_news.csv",
        PROJ_ROOT / "data/raw/news data/NFLX_financial_news.csv",
        PROJ_ROOT / "data/raw/news data/MSFT_financial_news.csv",
    ]

    for csv_path in dataset_paths:
        print(f"\n=== Processing: {csv_path} ===")
        source_name = Path(csv_path).stem

        analyzer = StockSentimentComparison(csv_path, source_name)
        results_df, agreement_matrix = analyzer.run_complete_analysis()
        

        results_file   = f"{source_name}_sentiment_analysis_results.csv"
        agree_file     = f"{source_name}_model_agreement_matrix.csv"

        results_df.to_csv(f"{PROJ_ROOT}/data/interim/{results_file}", index=False)
        agreement_matrix.to_csv(f"{PROJ_ROOT}/data/interim/{agree_file}")

        logger.info("→  Analysis complete!  Files saved:")
        logger.info(f"   – {results_file}")
        logger.info(f"   – {agree_file}")
        logger.info(f"     DataFrame shape: {results_df.shape}")
        logger.info("     Columns:")
        for col in results_df.columns:
            logger.info(f"       · {col}")
    

    # Added trained lr, svm and rf with the financial news
    dataset_paths = [
        PROJ_ROOT / "data/interim/AAPL_financial_news_sentiment_analysis_results.csv",
        PROJ_ROOT / "data/interim/TSLA_financial_news_sentiment_analysis_results.csv",
        PROJ_ROOT / "data/interim/AMZN_financial_news_sentiment_analysis_results.csv",
        PROJ_ROOT / "data/interim/NFLX_financial_news_sentiment_analysis_results.csv",
        PROJ_ROOT / "data/interim/MSFT_financial_news_sentiment_analysis_results.csv",
    ]
    

    for csv_path in dataset_paths:
        print(f"\n=== Processing: {csv_path} ===")
        source_name = Path(csv_path).stem

        df = pd.read_csv(csv_path)
        analyzer = StockSentimentComparison(path=None, ticker=source_name)
        df = analyzer.create_comparison_table(df, models_dir=str(PROJ_ROOT / "models"))

if __name__ == "__main__":
    main()