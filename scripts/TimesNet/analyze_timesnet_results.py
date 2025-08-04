#!/usr/bin/env python3
"""
TimesNet Results Analysis Script
Analyzes TXT files from TimesNet output and calculates mean metrics across 5 tickers
for each model and output type combination.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Define the constants
TICKERS = ['AAPL', 'AMZN', 'MSFT', 'NFLX', 'TSLA']
MODELS = ['deberta', 'finbert', 'lr', 'rf', 'roberta', 'svm']
OUTPUT_TYPES = ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']
TIMESNET_DIR = 'reports/output/TimesNet'

def parse_metrics_from_file(file_path):
    """
    Parse metrics from a TXT file.
    Returns the metrics dictionary from the last line (TimesNet format).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(lines) >= 1:
            # Get the last line which contains the metrics
            metrics_line = lines[-1].strip()
            # Extract the dictionary part after "Metrics: "
            if "Metrics: " in metrics_line:
                metrics_str = metrics_line.split("Metrics: ", 1)[1]
                
                # Handle inf and nan values by replacing them with actual float values
                # that ast.literal_eval can handle
                metrics_str = re.sub(r'\binf\b', 'float("inf")', metrics_str)
                metrics_str = re.sub(r'\bnan\b', 'float("nan")', metrics_str)
                
                # Since ast.literal_eval can't handle float() calls, we need to use eval instead
                # but restrict the allowed names for security
                allowed_names = {
                    "__builtins__": {},
                    "float": float,
                    "inf": float('inf'),
                    "nan": float('nan')
                }
                
                # Use eval with restricted globals for safety
                metrics = eval(metrics_str, allowed_names, {})
                return metrics
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    
    return None

def extract_file_info(filename):
    """
    Extract ticker, model, and output_type from filename.
    Expected format: {TICKER}_{MODEL}_{OUTPUT_TYPE}_pred_vs_true.txt
    """
    # Remove the extension and suffix
    base_name = filename.replace('_pred_vs_true.txt', '')
    parts = base_name.split('_')
    
    if len(parts) >= 3:
        ticker = parts[0]
        model = parts[1]
        output_type = '_'.join(parts[2:])  # Handle cases like "Binary_Price"
        return ticker, model, output_type
    
    return None, None, None

def analyze_classification_metrics(metrics_dict):
    """
    Extract key metrics from binary classification results.
    """
    if not metrics_dict:
        return {}
    
    result = {}
    
    # Extract basic metrics
    if 'auc' in metrics_dict:
        result['AUC'] = metrics_dict['auc']
    if 'accuracy' in metrics_dict:
        result['Accuracy'] = metrics_dict['accuracy']
    
    # Extract classification report metrics
    if 'classification_report' in metrics_dict:
        report = metrics_dict['classification_report']
        if 'weighted avg' in report:
            weighted_avg = report['weighted avg']
            result['Precision'] = weighted_avg.get('precision', None)
            result['Recall'] = weighted_avg.get('recall', None)
            result['F1-Score'] = weighted_avg.get('f1-score', None)
    
    return result

def analyze_regression_metrics(metrics_dict):
    """
    Extract key metrics from regression results.
    """
    if not metrics_dict:
        return {}
    
    result = {}
    
    # Common regression metrics
    metric_keys = ['RSE', 'CORR', 'MAE', 'MAPE', 'RMSE', 'MSLE']
    
    for key in metric_keys:
        if key in metrics_dict:
            value = metrics_dict[key]
            # Handle inf and nan values
            if value == float('inf') or value == float('-inf') or (isinstance(value, float) and np.isnan(value)):
                result[key] = None
            else:
                result[key] = value
        else:
            result[key] = None
    
    return result

def main():
    """
    Main analysis function.
    """
    # Create output file
    output_file = os.path.join(TIMESNET_DIR, "timesnet_analysis_results.txt")
    
    # Redirect output to both console and file
    import sys
    from io import StringIO
    
    # Capture output
    output_buffer = StringIO()
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to capture output
        sys.stdout = output_buffer
        
        print("=" * 80)
        print("TIMESNET RESULTS ANALYSIS")
        print("=" * 80)
        print()
        
        # Check if TimesNet directory exists
        if not os.path.exists(TIMESNET_DIR):
            print(f"Error: Directory {TIMESNET_DIR} does not exist!")
            return
        
        # Initialize data structures
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        # Find all TXT files
        txt_files = []
        for file in os.listdir(TIMESNET_DIR):
            if file.endswith('_pred_vs_true.txt'):
                txt_files.append(file)
        
        print(f"Found {len(txt_files)} TXT files to analyze...")
        print()
        
        # Parse each file
        for filename in txt_files:
            file_path = os.path.join(TIMESNET_DIR, filename)
            ticker, model, output_type = extract_file_info(filename)
            
            if ticker in TICKERS and model in MODELS and output_type in OUTPUT_TYPES:
                metrics = parse_metrics_from_file(file_path)
                if metrics:
                    if output_type == 'Binary_Price':
                        processed_metrics = analyze_classification_metrics(metrics)
                    else:
                        processed_metrics = analyze_regression_metrics(metrics)
                    
                    results[model][output_type][ticker] = processed_metrics
                    print(f"✓ Processed: {filename}")
                else:
                    print(f"✗ Failed to parse: {filename}")
            else:
                print(f"⚠ Skipped (invalid format): {filename}")
        
        print()
        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        # Calculate and display means for each model and output type
        for model in MODELS:
            print(f"MODEL: {model.upper()}")
            print("-" * 60)
            
            for output_type in OUTPUT_TYPES:
                print(f"\n  Output Type: {output_type}")
                print("  " + "=" * 50)
                
                # Check how many tickers we have data for
                available_tickers = []
                for ticker in TICKERS:
                    if ticker in results[model][output_type] and results[model][output_type][ticker]:
                        available_tickers.append(ticker)
                
                if not available_tickers:
                    print(f"    No data available for {model} - {output_type}")
                    continue
                
                print(f"    Available tickers: {', '.join(available_tickers)} ({len(available_tickers)}/5)")
                
                # Get all metric names from the first available ticker
                first_ticker = available_tickers[0]
                metric_names = list(results[model][output_type][first_ticker].keys())
                
                if not metric_names:
                    print(f"    No metrics found for {model} - {output_type}")
                    continue
                
                # Calculate means for each metric
                print(f"    \n    {'Metric':<15} {'Mean':<12} {'Count':<8} {'Tickers with data'}")
                print(f"    {'-'*15} {'-'*12} {'-'*8} {'-'*20}")
                
                for metric in metric_names:
                    values = []
                    valid_tickers = []
                    
                    for ticker in available_tickers:
                        value = results[model][output_type][ticker].get(metric)
                        if value is not None and not (isinstance(value, float) and np.isnan(value)):
                            values.append(value)
                            valid_tickers.append(ticker)
                    
                    if values:
                        mean_value = np.mean(values)
                        print(f"    {metric:<15} {mean_value:<12.4f} {len(values):<8} {', '.join(valid_tickers)}")
                    else:
                        print(f"    {metric:<15} {'N/A':<12} {0:<8} None")
            
            print("\n")
        
        # Summary statistics
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print()
        
        # Count available combinations
        total_combinations = len(MODELS) * len(OUTPUT_TYPES)
        available_combinations = 0
        
        for model in MODELS:
            for output_type in OUTPUT_TYPES:
                if any(results[model][output_type][ticker] for ticker in TICKERS):
                    available_combinations += 1
        
        print(f"Total possible combinations: {total_combinations}")
        print(f"Available combinations: {available_combinations}")
        print(f"Coverage: {available_combinations/total_combinations*100:.1f}%")
        print()
        
        # Model coverage
        print("Model Coverage:")
        for model in MODELS:
            model_coverage = 0
            for output_type in OUTPUT_TYPES:
                if any(results[model][output_type][ticker] for ticker in TICKERS):
                    model_coverage += 1
            print(f"  {model:<10}: {model_coverage}/{len(OUTPUT_TYPES)} output types")
        
        print()
        
        # Output type coverage
        print("Output Type Coverage:")
        for output_type in OUTPUT_TYPES:
            output_coverage = 0
            for model in MODELS:
                if any(results[model][output_type][ticker] for ticker in TICKERS):
                    output_coverage += 1
            print(f"  {output_type:<15}: {output_coverage}/{len(MODELS)} models")
        
        print()
        print("Analysis complete!")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        
        # Get the captured output
        output_content = output_buffer.getvalue()
        output_buffer.close()
        
        # Print to console
        print(output_content)
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_content)
            print(f"\nAnalysis results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

if __name__ == "__main__":
    main()