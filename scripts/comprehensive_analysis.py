#!/usr/bin/env python3
"""
Comprehensive Model Analysis Script
Analyzes all model outputs (LSTM, PatchTST, TimesNet, tPatchGNN) across all seeds
and sentiment models, providing comparisons based on mean performance across seeds.
"""

import os
import re
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
import json
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Define the constants
TICKERS = ['AAPL', 'AMZN', 'MSFT', 'NFLX', 'TSLA']
SENTIMENT_MODELS = ['deberta', 'finbert', 'lr', 'rf', 'roberta', 'svm']
OUTPUT_TYPES = ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']
MODEL_TYPES = ['LSTM', 'PatchTST', 'TimesNet', 'tPatchGNN']
SEEDS = ['16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192']
BASE_OUTPUT_DIR = 'reports/output'

def parse_metrics_from_file(file_path: str) -> Optional[Dict]:
    """
    Parse metrics from a TXT file.
    Returns the metrics dictionary from the line containing "Metrics: ".
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Find the line containing "Metrics: "
        metrics_line = None
        for line in lines:
            if "Metrics: " in line:
                metrics_line = line.strip()
                break
        
        if metrics_line:
            # Extract the dictionary part after "Metrics: "
            metrics_str = metrics_line.split("Metrics: ", 1)[1]
            
            # Handle inf and nan values by replacing them with actual float values
            metrics_str = re.sub(r'\binf\b', 'float("inf")', metrics_str)
            metrics_str = re.sub(r'\bnan\b', 'float("nan")', metrics_str)
            
            # Use eval with restricted globals for safety
            allowed_names = {
                "__builtins__": {},
                "float": float,
                "inf": float('inf'),
                "nan": float('nan')
            }
            
            metrics = eval(metrics_str, allowed_names, {})
            return metrics
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    
    return None

def extract_file_info(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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

def analyze_classification_metrics(metrics_dict: Dict) -> Dict:
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

def analyze_regression_metrics(metrics_dict: Dict) -> Dict:
    """
    Extract key metrics from regression results.
    """
    if not metrics_dict:
        return {}
    
    result = {}
    
    # Extract regression metrics (handle both lowercase and uppercase variants)
    if 'mse' in metrics_dict:
        result['MSE'] = metrics_dict['mse']
    elif 'MSE' in metrics_dict:
        result['MSE'] = metrics_dict['MSE']
    
    if 'mae' in metrics_dict:
        result['MAE'] = metrics_dict['mae']
    elif 'MAE' in metrics_dict:
        result['MAE'] = metrics_dict['MAE']
    
    if 'rmse' in metrics_dict:
        result['RMSE'] = metrics_dict['rmse']
    elif 'RMSE' in metrics_dict:
        result['RMSE'] = metrics_dict['RMSE']
    
    if 'mape' in metrics_dict:
        result['MAPE'] = metrics_dict['mape']
    elif 'MAPE' in metrics_dict:
        result['MAPE'] = metrics_dict['MAPE']
    
    if 'r2' in metrics_dict:
        result['R2'] = metrics_dict['r2']
    elif 'R2' in metrics_dict:
        result['R2'] = metrics_dict['R2']
    
    # Handle additional metrics that might be present
    if 'rse' in metrics_dict:
        result['RSE'] = metrics_dict['rse']
    elif 'RSE' in metrics_dict:
        result['RSE'] = metrics_dict['RSE']
    
    if 'corr' in metrics_dict:
        result['CORR'] = metrics_dict['corr']
    elif 'CORR' in metrics_dict:
        result['CORR'] = metrics_dict['CORR']
    
    if 'msle' in metrics_dict:
        result['MSLE'] = metrics_dict['msle']
    elif 'MSLE' in metrics_dict:
        result['MSLE'] = metrics_dict['MSLE']
    
    return result

def collect_all_results() -> Dict:
    """
    Collect results from all model types, seeds, sentiment models, and output types.
    """
    all_results = {}
    
    for model_type in MODEL_TYPES:
        model_dir = os.path.join(BASE_OUTPUT_DIR, model_type)
        if not os.path.exists(model_dir):
            print(f"Warning: Directory {model_dir} does not exist!")
            continue
            
        print(f"Processing {model_type}...")
        all_results[model_type] = {}
        
        # Check if there are seed subdirectories
        seed_dirs = [d for d in os.listdir(model_dir) if d.isdigit() and os.path.isdir(os.path.join(model_dir, d))]
        
        if seed_dirs:
            print(f"  Found {len(seed_dirs)} seed directories: {seed_dirs}")
            # Process seed-based results
            for seed in seed_dirs:
                if seed not in SEEDS:
                    continue
                    
                seed_dir = os.path.join(model_dir, seed)
                all_results[model_type][seed] = {}
                
                # Find all TXT files in this seed directory
                txt_files = [f for f in os.listdir(seed_dir) if f.endswith('_pred_vs_true.txt')]
                print(f"    Seed {seed}: Found {len(txt_files)} TXT files")
                
                processed_count = 0
                for filename in txt_files:
                    file_path = os.path.join(seed_dir, filename)
                    ticker, sentiment_model, output_type = extract_file_info(filename)
                    
                    if (ticker in TICKERS and 
                        sentiment_model in SENTIMENT_MODELS and 
                        output_type in OUTPUT_TYPES):
                        
                        metrics = parse_metrics_from_file(file_path)
                        if metrics:
                            if output_type == 'Binary_Price':
                                processed_metrics = analyze_classification_metrics(metrics)
                            else:
                                processed_metrics = analyze_regression_metrics(metrics)
                            
                            if sentiment_model not in all_results[model_type][seed]:
                                all_results[model_type][seed][sentiment_model] = {}
                            if output_type not in all_results[model_type][seed][sentiment_model]:
                                all_results[model_type][seed][sentiment_model][output_type] = {}
                            
                            all_results[model_type][seed][sentiment_model][output_type][ticker] = processed_metrics
                            processed_count += 1
                
                print(f"    Seed {seed}: Successfully processed {processed_count} files")
        else:
            print(f"  No seed directories found, checking main directory")
            # Process results directly in model directory (no seed subdirectories)
            all_results[model_type]['main'] = {}
            
            txt_files = [f for f in os.listdir(model_dir) if f.endswith('_pred_vs_true.txt')]
            print(f"  Found {len(txt_files)} TXT files in main directory")
            
            processed_count = 0
            for filename in txt_files:
                file_path = os.path.join(model_dir, filename)
                ticker, sentiment_model, output_type = extract_file_info(filename)
                
                if (ticker in TICKERS and 
                    sentiment_model in SENTIMENT_MODELS and 
                    output_type in OUTPUT_TYPES):
                    
                    metrics = parse_metrics_from_file(file_path)
                    if metrics:
                        if output_type == 'Binary_Price':
                            processed_metrics = analyze_classification_metrics(metrics)
                        else:
                            processed_metrics = analyze_regression_metrics(metrics)
                        
                        if sentiment_model not in all_results[model_type]['main']:
                            all_results[model_type]['main'][sentiment_model] = {}
                        if output_type not in all_results[model_type]['main'][sentiment_model]:
                            all_results[model_type]['main'][sentiment_model][output_type] = {}
                        
                        all_results[model_type]['main'][sentiment_model][output_type][ticker] = processed_metrics
                        processed_count += 1
            
            print(f"  Successfully processed {processed_count} files from main directory")
    
    return all_results

def calculate_seed_means(all_results: Dict) -> Dict:
    """
    Calculate mean metrics across seeds for each model type, sentiment model, and output type.
    """
    seed_means = {}
    
    for model_type in MODEL_TYPES:
        if model_type not in all_results:
            continue
            
        seed_means[model_type] = {}
        
        # Get all available seeds for this model type
        available_seeds = [seed for seed in all_results[model_type].keys() if seed != 'main']
        
        if not available_seeds:
            # No seed-based results, use main results if available
            if 'main' in all_results[model_type]:
                seed_means[model_type] = all_results[model_type]['main']
            continue
        
        # Calculate means across seeds
        for sentiment_model in SENTIMENT_MODELS:
            if sentiment_model not in seed_means[model_type]:
                seed_means[model_type][sentiment_model] = {}
                
            for output_type in OUTPUT_TYPES:
                if output_type not in seed_means[model_type][sentiment_model]:
                    seed_means[model_type][sentiment_model][output_type] = {}
                
                # Collect all metrics for this combination across seeds
                ticker_metrics = defaultdict(list)
                
                for seed in available_seeds:
                    if (sentiment_model in all_results[model_type][seed] and
                        output_type in all_results[model_type][seed][sentiment_model]):
                        
                        for ticker in TICKERS:
                            if ticker in all_results[model_type][seed][sentiment_model][output_type]:
                                metrics = all_results[model_type][seed][sentiment_model][output_type][ticker]
                                ticker_metrics[ticker].append(metrics)
                
                # Calculate means for each ticker
                for ticker in TICKERS:
                    if ticker in ticker_metrics and ticker_metrics[ticker]:
                        # Get all metric names from the first result
                        metric_names = list(ticker_metrics[ticker][0].keys())
                        mean_metrics = {}
                        
                        for metric in metric_names:
                            values = []
                            for seed_metrics in ticker_metrics[ticker]:
                                if metric in seed_metrics and seed_metrics[metric] is not None:
                                    value = seed_metrics[metric]
                                    if not (isinstance(value, float) and np.isnan(value)):
                                        values.append(value)
                            
                            if values:
                                mean_metrics[metric] = np.mean(values)
                        
                        if mean_metrics:
                            seed_means[model_type][sentiment_model][output_type][ticker] = mean_metrics
    
    return seed_means

def generate_comparison_report(seed_means: Dict) -> str:
    """
    Generate a comprehensive comparison report.
    """
    report_lines = []
    
    report_lines.append("=" * 100)
    report_lines.append("COMPREHENSIVE MODEL ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("This report compares all model types across sentiment models and output types")
    report_lines.append("based on mean performance across seeds.")
    report_lines.append("")
    
    # Overall summary
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 50)
    
    for model_type in MODEL_TYPES:
        if model_type in seed_means:
            model_data = seed_means[model_type]
            sentiment_count = len(model_data)
            total_combinations = 0
            available_combinations = 0
            
            for sentiment_model in SENTIMENT_MODELS:
                if sentiment_model in model_data:
                    for output_type in OUTPUT_TYPES:
                        total_combinations += 1
                        if output_type in model_data[sentiment_model]:
                            ticker_count = len(model_data[sentiment_model][output_type])
                            if ticker_count > 0:
                                available_combinations += 1
            
            coverage = (available_combinations / total_combinations * 100) if total_combinations > 0 else 0
            report_lines.append(f"{model_type:<12}: {sentiment_count}/{len(SENTIMENT_MODELS)} sentiment models, "
                              f"{available_combinations}/{total_combinations} combinations ({coverage:.1f}% coverage)")
    
    report_lines.append("")
    
    # Detailed analysis by output type
    for output_type in OUTPUT_TYPES:
        report_lines.append(f"OUTPUT TYPE: {output_type}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Create comparison table
        metric_names = set()
        for model_type in MODEL_TYPES:
            if model_type in seed_means:
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in seed_means[model_type] and
                        output_type in seed_means[model_type][sentiment_model]):
                        for ticker in TICKERS:
                            if ticker in seed_means[model_type][sentiment_model][output_type]:
                                metrics = seed_means[model_type][sentiment_model][output_type][ticker]
                                metric_names.update(metrics.keys())
        
        if not metric_names:
            report_lines.append("No data available for this output type.")
            report_lines.append("")
            continue
        
        metric_names = sorted(list(metric_names))
        
        # Header
        header = f"{'Model':<12} {'Sentiment':<10} {'Ticker':<6}"
        for metric in metric_names:
            header += f" {metric:<10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        # Data rows
        for model_type in MODEL_TYPES:
            if model_type not in seed_means:
                continue
                
            for sentiment_model in SENTIMENT_MODELS:
                if (sentiment_model not in seed_means[model_type] or
                    output_type not in seed_means[model_type][sentiment_model]):
                    continue
                
                for ticker in TICKERS:
                    if ticker not in seed_means[model_type][sentiment_model][output_type]:
                        continue
                    
                    metrics = seed_means[model_type][sentiment_model][output_type][ticker]
                    row = f"{model_type:<12} {sentiment_model:<10} {ticker:<6}"
                    
                    for metric in metric_names:
                        value = metrics.get(metric, 'N/A')
                        if isinstance(value, float):
                            row += f" {value:<10.4f}"
                        else:
                            row += f" {str(value):<10}"
                    
                    report_lines.append(row)
        
        report_lines.append("")
        
        # Best performing models for each metric
        report_lines.append("BEST PERFORMING MODELS (by metric):")
        report_lines.append("-" * 50)
        
        for metric in metric_names:
            best_value = None
            best_combination = None
            
            for model_type in MODEL_TYPES:
                if model_type not in seed_means:
                    continue
                    
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model not in seed_means[model_type] or
                        output_type not in seed_means[model_type][sentiment_model]):
                        continue
                    
                    for ticker in TICKERS:
                        if ticker not in seed_means[model_type][sentiment_model][output_type]:
                            continue
                        
                        metrics = seed_means[model_type][sentiment_model][output_type][ticker]
                        if metric in metrics:
                            value = metrics[metric]
                            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                if best_value is None:
                                    best_value = value
                                    best_combination = f"{model_type}-{sentiment_model}-{ticker}"
                                else:
                                    # For classification metrics, higher is better
                                    if metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
                                        if value > best_value:
                                            best_value = value
                                            best_combination = f"{model_type}-{sentiment_model}-{ticker}"
                                    # For regression metrics, lower is better (except CORR)
                                    elif metric == 'CORR':
                                        if value > best_value:
                                            best_value = value
                                            best_combination = f"{model_type}-{sentiment_model}-{ticker}"
                                    else:  # All other regression error metrics
                                        if value < best_value:
                                            best_value = value
                                            best_combination = f"{model_type}-{sentiment_model}-{ticker}"
            
            if best_combination:
                report_lines.append(f"{metric:<15}: {best_combination:<30} ({best_value:.4f})")
        
        report_lines.append("")
    
    return "\n".join(report_lines)

def save_detailed_results(seed_means: Dict, output_file: str):
    """
    Save detailed results to a JSON file for further analysis.
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    converted_data = convert_numpy_types(seed_means)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to: {output_file}")

def main():
    """
    Main analysis function.
    """
    print("Starting comprehensive model analysis...")
    print("=" * 80)
    
    # Collect all results
    print("Collecting results from all model types and seeds...")
    all_results = collect_all_results()
    
    # Calculate seed means
    print("Calculating mean metrics across seeds...")
    seed_means = calculate_seed_means(all_results)
    
    # Generate comparison report
    print("Generating comparison report...")
    report = generate_comparison_report(seed_means)
    
    # Save results
    output_dir = "reports/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text report
    report_file = os.path.join(output_dir, "comprehensive_analysis_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save detailed JSON results
    json_file = os.path.join(output_dir, "comprehensive_analysis_results.json")
    save_detailed_results(seed_means, json_file)
    
    # Print report to console
    print("\n" + report)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {report_file}")
    print(f"Detailed results saved to: {json_file}")

if __name__ == "__main__":
    main()
