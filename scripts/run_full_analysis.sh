#!/bin/bash

# Full Analysis and Visualization Script
# This script runs the comprehensive analysis and generates visualizations

echo "=========================================="
echo "Full Model Analysis and Visualization"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/comprehensive_analysis.py" ]; then
    echo "Error: comprehensive_analysis.py not found in scripts directory!"
    echo "Please run this script from the project root directory."
    exit 1
fi

if [ ! -f "scripts/generate_visualizations.py" ]; then
    echo "Error: generate_visualizations.py not found in scripts directory!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if output directory exists
if [ ! -d "reports/output" ]; then
    echo "Error: reports/output directory not found!"
    echo "Please ensure the output directory exists with model results."
    exit 1
fi

echo "Step 1: Running comprehensive analysis..."
echo "This will analyze all model outputs across:"
echo "- Model types: LSTM, PatchTST, TimesNet, tPatchGNN"
echo "- Sentiment models: deberta, finbert, lr, rf, roberta, svm"
echo "- Output types: Binary_Price, Delta_Price, Factor_Price, Float_Price"
echo "- Seeds: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192"
echo "- Tickers: AAPL, AMZN, MSFT, NFLX, TSLA"
echo ""

# Run the comprehensive analysis
python scripts/comprehensive_analysis.py

if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis failed! Stopping execution."
    echo "=========================================="
    exit 1
fi

echo ""
echo "Step 2: Generating visualizations..."
echo "This will create:"
echo "- Heatmaps comparing models across metrics"
echo "- Bar plots for metric comparisons"
echo "- Summary plots of overall performance"
echo ""

# Run the visualization generation
python scripts/generate_visualizations.py

if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "Visualization generation failed!"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Full analysis completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo ""
echo "Analysis Results:"
echo "- reports/output/comprehensive_analysis_report.txt"
echo "- reports/output/comprehensive_analysis_results.json"
echo ""
echo "Visualizations:"
echo "- reports/output/visualizations/"

# List visualization files if they exist
if [ -d "reports/output/visualizations" ]; then
    echo ""
    echo "Visualization files:"
    for file in reports/output/visualizations/*.png; do
        if [ -f "$file" ]; then
            echo "  - $(basename "$file")"
        fi
    done
fi

echo ""
echo "You can view the detailed report with:"
echo "cat reports/output/comprehensive_analysis_report.txt"
echo ""
echo "You can view the visualizations in:"
echo "reports/output/visualizations/"
