from FinancialSentimentComparison import FinancialSentimentComparison
from pathlib import Path
from loguru import logger
from stock_prediction.config import *
from ModelTrainer import ModelTrainer
import pandas as pd

def main():
    # Run the sentiment analysis of Roberta, FinBERT and DeBERTa models with SentFiN-v1.1 dataset and save the results
    dataset_paths = [
        PROJ_ROOT / "data/raw/human_labeled/SEntFiN-v1.1.csv",
    ]

    for csv_path in dataset_paths:
        logger.info(f"\n=== Processing: {csv_path} ===")
        source_name = Path(csv_path).stem

        analyzer = FinancialSentimentComparison(csv_path, "")
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
    model_trainer = ModelTrainer(results_df, x_cols, target_col)

    model_trainer.train_logistic_regression()
    model_trainer.train_svm()
    model_trainer.train_random_forest()

if __name__ == "__main__":
    main()