#!/bin/bash

# Comprehensive Model Analysis Script
# This script runs the comprehensive analysis across all model types and seeds

echo "=========================================="
echo "Comprehensive Model Analysis"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/comprehensive_analysis.py" ]; then
    echo "Error: comprehensive_analysis.py not found in scripts directory!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if output directory exists
if [ ! -d "reports/output" ]; then
    echo "Error: reports/output directory not found!"
    echo "Please ensure the output directory exists with model results."
    exit 1
fi

echo "Starting comprehensive analysis..."
echo "This will analyze all model outputs across:"
echo "- Model types: LSTM, PatchTST, TimesNet, tPatchGNN"
echo "- Sentiment models: deberta, finbert, lr, rf, roberta, svm"
echo "- Output types: Binary_Price, Delta_Price, Factor_Price, Float_Price"
echo "- Seeds: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192"
echo "- Tickers: AAPL, AMZN, MSFT, NFLX, TSLA"
echo ""

# Run the comprehensive analysis
python scripts/comprehensive_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files generated:"
    echo "- reports/output/comprehensive_analysis_report.txt"
    echo "- reports/output/comprehensive_analysis_results.json"
    echo ""
    echo "The report contains:"
    echo "- Overall summary of model coverage"
    echo "- Detailed comparison tables by output type"
    echo "- Best performing models for each metric"
    echo "- Mean performance across all seeds"
    echo ""
    echo "You can view the detailed report with:"
    echo "cat reports/output/comprehensive_analysis_report.txt"
else
    echo ""
    echo "=========================================="
    echo "Analysis failed!"
    echo "=========================================="
    echo "Please check the error messages above."
    exit 1
fi
