# Comprehensive Model Analysis Scripts

This directory contains scripts for comprehensive analysis of all model outputs across different model types, sentiment models, seeds, and output types.

## Overview

The analysis covers:
- **Model Types**: LSTM, PatchTST, TimesNet, tPatchGNN
- **Sentiment Models**: deberta, finbert, lr, rf, roberta, svm
- **Output Types**: Binary_Price, Delta_Price, Factor_Price, Float_Price
- **Seeds**: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
- **Tickers**: AAPL, AMZN, MSFT, NFLX, TSLA

## Scripts

### 1. `comprehensive_analysis.py`
Main analysis script that:
- Collects results from all model outputs across seeds
- Calculates mean metrics across seeds for each combination
- Generates a comprehensive comparison report
- Saves detailed results in JSON format

**Outputs:**
- `reports/output/comprehensive_analysis_report.txt` - Detailed text report
- `reports/output/comprehensive_analysis_results.json` - Structured data for further analysis

### 2. `generate_visualizations.py`
Visualization script that creates:
- Heatmaps comparing models across metrics
- Bar plots for metric comparisons by model type
- Summary plots of overall performance

**Outputs:**
- `reports/output/visualizations/` - Directory containing all visualization files

### 3. `generate_aggregated_visualizations.py`
Aggregated visualization script that creates cleaner plots by averaging across all 5 tickers:
- Heatmaps comparing models vs sentiment models (averaged across tickers)
- Bar plots for metric comparisons by model type (averaged across tickers)
- Summary plots of overall performance (averaged across tickers)
- Sentiment model comparison plots (averaged across tickers)

**Outputs:**
- `reports/output/aggregated_visualizations/` - Directory containing all aggregated visualization files

### 4. `run_comprehensive_analysis.sh` (Linux/Mac)
Shell script to run only the comprehensive analysis.

### 5. `run_full_analysis.sh` (Linux/Mac)
Shell script to run both analysis and visualization generation.

### 6. `run_full_analysis.ps1` (Windows)
PowerShell script to run both analysis and visualization generation.

### 7. `run_aggregated_visualizations.ps1` (Windows)
PowerShell script to run only the aggregated visualization generation.

## Usage

### Option 1: Run Analysis Only
```bash
# Linux/Mac
./scripts/run_comprehensive_analysis.sh

# Windows
python scripts/comprehensive_analysis.py
```

### Option 2: Run Full Analysis with Visualizations
```bash
# Linux/Mac
./scripts/run_full_analysis.sh

# Windows
powershell -ExecutionPolicy Bypass -File scripts/run_full_analysis.ps1
```

### Option 3: Run Individual Scripts
```bash
# Run analysis
python scripts/comprehensive_analysis.py

# Run visualizations (requires analysis results)
python scripts/generate_visualizations.py

# Run aggregated visualizations (requires analysis results)
python scripts/generate_aggregated_visualizations.py
```

### Option 4: Run Aggregated Visualizations Only (Windows)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_aggregated_visualizations.ps1
```

## Output Structure

### Text Report (`comprehensive_analysis_report.txt`)
The report contains:
1. **Overall Summary**: Coverage statistics for each model type
2. **Detailed Analysis by Output Type**: 
   - Comparison tables with all metrics
   - Best performing models for each metric
3. **Summary Statistics**: Overall coverage and availability

### JSON Results (`comprehensive_analysis_results.json`)
Structured data organized as:
```json
{
  "LSTM": {
    "deberta": {
      "Binary_Price": {
        "AAPL": {
          "AUC": 0.75,
          "Accuracy": 0.68,
          "F1-Score": 0.72
        }
      }
    }
  }
}
```

### Visualizations (`visualizations/`)
- `heatmap_{output_type}_{metric}.png` - Heatmaps comparing models
- `barplot_{model_type}_{output_type}.png` - Bar plots for specific models
- `model_performance_summary.png` - Overall performance summary

### Aggregated Visualizations (`aggregated_visualizations/`)
- `heatmap_{output_type}_{metric}_aggregated.png` - Heatmaps comparing models vs sentiment models (averaged across tickers)
- `barplot_{model_type}_{output_type}_aggregated.png` - Bar plots for specific models (averaged across tickers)
- `model_performance_summary_aggregated.png` - Overall performance summary (averaged across tickers)
- `sentiment_comparison_{output_type}_aggregated.png` - Sentiment model comparison plots (averaged across tickers)

## Metrics Analyzed

### Classification Metrics (Binary_Price)
- AUC (Area Under Curve)
- Accuracy
- Precision
- Recall
- F1-Score

### Regression Metrics (Delta_Price, Factor_Price, Float_Price)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- RSE (Relative Squared Error)
- CORR (Correlation Coefficient)
- MAPE (Mean Absolute Percentage Error)
- MSLE (Mean Squared Logarithmic Error)

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- json (built-in)
- pathlib (built-in)

## Notes

1. **Seed Handling**: The script automatically detects if results are organized by seeds (subdirectories) or directly in model directories.

2. **Missing Data**: The analysis handles missing data gracefully and reports coverage statistics.

3. **Performance**: For large datasets, the analysis may take several minutes to complete.

4. **Memory Usage**: The script loads all results into memory, so ensure sufficient RAM for large datasets.

## Troubleshooting

### Common Issues

1. **"Directory not found"**: Ensure you're running from the project root directory and that `reports/output/` exists.

2. **"No data available"**: Check that the output files follow the expected naming convention: `{TICKER}_{SENTIMENT_MODEL}_{OUTPUT_TYPE}_pred_vs_true.txt`

3. **Import errors**: Install required packages with `pip install pandas numpy matplotlib seaborn`

4. **Permission errors** (Linux/Mac): Make scripts executable with `chmod +x scripts/*.sh`

### File Naming Convention
Expected format for result files:
```
{TICKER}_{SENTIMENT_MODEL}_{OUTPUT_TYPE}_pred_vs_true.txt
```

Examples:
- `AAPL_deberta_Binary_Price_pred_vs_true.txt`
- `MSFT_finbert_Delta_Price_pred_vs_true.txt`
- `TSLA_svm_Factor_Price_pred_vs_true.txt`
