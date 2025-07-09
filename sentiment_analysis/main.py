from FinancialSentimentComparison import FinancialSentimentComparison
from pathlib import Path
from loguru import logger
from stock_prediction.config import *

def main():
    
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


if __name__ == "__main__":
    main()