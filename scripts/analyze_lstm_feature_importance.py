#!/usr/bin/env python3
"""
LSTM Feature Importance Analysis Script
Analyzes and compares performance across three LSTM variants:
- LSTM: Full feature set including count features
- LSTM_wo_count: Without count features (svm_count_*)
- LSTM_wo_count_sum: Full feature set (same as LSTM)

This script helps understand the impact of different feature sets on model performance.
"""

import os
import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Define the constants
TICKERS = ['AAPL', 'AMZN', 'MSFT', 'NFLX', 'TSLA']
SENTIMENT_MODELS = ['deberta', 'finbert', 'lr', 'rf', 'roberta', 'svm']
OUTPUT_TYPES = ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']
LSTM_VARIANTS = ['LSTM', 'LSTM_wo_count', 'LSTM_wo_count_sum', 'LSTM_wo_sum','LSTM_wo_majority']
BASE_DIR = 'reports/output'

def parse_metrics_from_file(file_path):
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

def get_feature_columns(file_path):
    """
    Extract feature columns from a TXT file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            if "Feature Columns: " in line:
                # Extract the list part after "Feature Columns: "
                features_str = line.split("Feature Columns: ", 1)[1].strip()
                # Parse the list
                features = ast.literal_eval(features_str)
                return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
    
    return None

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
    
    # Common regression metrics (handle both lowercase and uppercase)
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

def collect_all_results():
    """
    Collect results from all LSTM variants.
    """
    all_results = {}
    
    for variant in LSTM_VARIANTS:
        variant_dir = os.path.join(BASE_DIR, variant)
        
        if not os.path.exists(variant_dir):
            print(f"Warning: Directory {variant_dir} does not exist!")
            continue
            
        print(f"Processing {variant}...")
        all_results[variant] = {}
        
        # Check if there are seed subdirectories
        seed_dirs = [d for d in os.listdir(variant_dir) if d.isdigit() and os.path.isdir(os.path.join(variant_dir, d))]
        
        if seed_dirs:
            print(f"  Found {len(seed_dirs)} seed directories")
            # Process seed-based results
            for seed in seed_dirs:
                seed_dir = os.path.join(variant_dir, seed)
                all_results[variant][seed] = {}
                
                # Find all TXT files in this seed directory
                txt_files = [f for f in os.listdir(seed_dir) if f.endswith('_pred_vs_true.txt')]
                
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
                            
                            # Get feature columns
                            feature_columns = get_feature_columns(file_path)
                            
                            if sentiment_model not in all_results[variant][seed]:
                                all_results[variant][seed][sentiment_model] = {}
                            if output_type not in all_results[variant][seed][sentiment_model]:
                                all_results[variant][seed][sentiment_model][output_type] = {}
                            
                            all_results[variant][seed][sentiment_model][output_type][ticker] = {
                                'metrics': processed_metrics,
                                'features': feature_columns
                            }
        else:
            print(f"  No seed directories found, checking main directory")
            # Process results directly in model directory (no seed subdirectories)
            all_results[variant]['main'] = {}
            
            txt_files = [f for f in os.listdir(variant_dir) if f.endswith('_pred_vs_true.txt')]
            
            for filename in txt_files:
                file_path = os.path.join(variant_dir, filename)
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
                        
                        # Get feature columns
                        feature_columns = get_feature_columns(file_path)
                        
                        if sentiment_model not in all_results[variant]['main']:
                            all_results[variant]['main'][sentiment_model] = {}
                        if output_type not in all_results[variant]['main'][sentiment_model]:
                            all_results[variant]['main'][sentiment_model][output_type] = {}
                        
                        all_results[variant]['main'][sentiment_model][output_type][ticker] = {
                            'metrics': processed_metrics,
                            'features': feature_columns
                        }
    
    return all_results

def calculate_variant_means(all_results):
    """
    Calculate mean metrics across all seeds and tickers for each variant.
    """
    variant_means = {}
    
    for variant in LSTM_VARIANTS:
        if variant not in all_results:
            continue
            
        variant_means[variant] = {}
        
        for sentiment_model in SENTIMENT_MODELS:
            variant_means[variant][sentiment_model] = {}
            
            for output_type in OUTPUT_TYPES:
                variant_means[variant][sentiment_model][output_type] = {}
                
                # Collect all metrics for this combination
                all_metrics = {}
                all_features = []
                
                # Iterate through seeds
                for seed_key in all_results[variant]:
                    if sentiment_model in all_results[variant][seed_key]:
                        if output_type in all_results[variant][seed_key][sentiment_model]:
                            for ticker in TICKERS:
                                if ticker in all_results[variant][seed_key][sentiment_model][output_type]:
                                    data = all_results[variant][seed_key][sentiment_model][output_type][ticker]
                                    metrics = data['metrics']
                                    features = data['features']
                                    
                                    if features:
                                        all_features.append(features)
                                    
                                    for metric, value in metrics.items():
                                        if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                            if metric not in all_metrics:
                                                all_metrics[metric] = []
                                            all_metrics[metric].append(value)
                
                # Calculate means for each metric
                mean_metrics = {}
                for metric, values in all_metrics.items():
                    if values:
                        mean_metrics[metric] = np.mean(values)
                
                if mean_metrics:
                    variant_means[variant][sentiment_model][output_type] = {
                        'metrics': mean_metrics,
                        'feature_count': len(all_features[0]) if all_features else 0,
                        'features': all_features[0] if all_features else []
                    }
    
    return variant_means

def generate_comparison_report(variant_means):
    """
    Generate a comprehensive comparison report.
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("LSTM FEATURE IMPORTANCE ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("This analysis compares three LSTM variants:")
    report_lines.append("- LSTM: Full feature set including count features")
    report_lines.append("- LSTM_wo_count: Without count features (svm_count_*)")
    report_lines.append("- LSTM_wo_count_sum: Full feature set (same as LSTM)")
    report_lines.append("- LSTM_wo_sum: Full feature set (same as LSTM)")
    report_lines.append("")
    
    # Feature set comparison
    report_lines.append("FEATURE SET COMPARISON")
    report_lines.append("-" * 40)
    report_lines.append("")
    
    for variant in LSTM_VARIANTS:
        if variant in variant_means:
            # Get sample features from first available combination
            sample_features = None
            for sentiment_model in SENTIMENT_MODELS:
                for output_type in OUTPUT_TYPES:
                    if (sentiment_model in variant_means[variant] and 
                        output_type in variant_means[variant][sentiment_model]):
                        sample_features = variant_means[variant][sentiment_model][output_type]['features']
                        break
                if sample_features:
                    break
            
            if sample_features:
                report_lines.append(f"{variant}:")
                report_lines.append(f"  Features ({len(sample_features)}): {sample_features}")
                report_lines.append("")
    
    # Performance comparison by output type
    for output_type in OUTPUT_TYPES:
        report_lines.append(f"PERFORMANCE COMPARISON - {output_type}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Get all available metrics for this output type
        all_metrics = set()
        for variant in LSTM_VARIANTS:
            if variant in variant_means:
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in variant_means[variant] and 
                        output_type in variant_means[variant][sentiment_model]):
                        metrics = variant_means[variant][sentiment_model][output_type]['metrics']
                        all_metrics.update(metrics.keys())
        
        if not all_metrics:
            report_lines.append(f"No data available for {output_type}")
            report_lines.append("")
            continue
        
        all_metrics = sorted(list(all_metrics))
        
        # Create comparison table
        header = f"{'Variant':<15} {'Sentiment':<10}"
        for metric in all_metrics:
            header += f" {metric:<10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for variant in LSTM_VARIANTS:
            if variant not in variant_means:
                continue
                
            for sentiment_model in SENTIMENT_MODELS:
                if (sentiment_model in variant_means[variant] and 
                    output_type in variant_means[variant][sentiment_model]):
                    
                    metrics = variant_means[variant][sentiment_model][output_type]['metrics']
                    row = f"{variant:<15} {sentiment_model:<10}"
                    
                    for metric in all_metrics:
                        value = metrics.get(metric, 'N/A')
                        if isinstance(value, (int, float)):
                            row += f" {value:<10.4f}"
                        else:
                            row += f" {value:<10}"
                    
                    report_lines.append(row)
        
        report_lines.append("")
        
        # Best performing variant for each metric
        report_lines.append("BEST PERFORMING VARIANT BY METRIC:")
        report_lines.append("-" * 40)
        
        for metric in all_metrics:
            best_variant = None
            best_value = None
            best_sentiment = None
            
            for variant in LSTM_VARIANTS:
                if variant not in variant_means:
                    continue
                    
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in variant_means[variant] and 
                        output_type in variant_means[variant][sentiment_model]):
                        
                        value = variant_means[variant][sentiment_model][output_type]['metrics'].get(metric)
                        if value is not None:
                            if best_value is None:
                                best_variant = variant
                                best_value = value
                                best_sentiment = sentiment_model
                            else:
                                # For classification metrics, higher is better
                                if metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
                                    if value > best_value:
                                        best_variant = variant
                                        best_value = value
                                        best_sentiment = sentiment_model
                                # For regression metrics, lower is better (except CORR)
                                elif metric == 'CORR':
                                    if value > best_value:
                                        best_variant = variant
                                        best_value = value
                                        best_sentiment = sentiment_model
                                else:
                                    if value < best_value:
                                        best_variant = variant
                                        best_value = value
                                        best_sentiment = sentiment_model
            
            if best_variant:
                report_lines.append(f"{metric:<15}: {best_variant} ({best_sentiment}) = {best_value:.4f}")
        
        report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("=" * 40)
    report_lines.append("")
    
    # Count available combinations
    total_combinations = len(LSTM_VARIANTS) * len(SENTIMENT_MODELS) * len(OUTPUT_TYPES)
    available_combinations = 0
    
    for variant in LSTM_VARIANTS:
        if variant in variant_means:
            for sentiment_model in SENTIMENT_MODELS:
                for output_type in OUTPUT_TYPES:
                    if (sentiment_model in variant_means[variant] and 
                        output_type in variant_means[variant][sentiment_model]):
                        available_combinations += 1
    
    report_lines.append(f"Total possible combinations: {total_combinations}")
    report_lines.append(f"Available combinations: {available_combinations}")
    report_lines.append(f"Coverage: {available_combinations/total_combinations*100:.1f}%")
    report_lines.append("")
    
    # Variant coverage
    report_lines.append("Variant Coverage:")
    for variant in LSTM_VARIANTS:
        if variant in variant_means:
            variant_coverage = 0
            for sentiment_model in SENTIMENT_MODELS:
                for output_type in OUTPUT_TYPES:
                    if (sentiment_model in variant_means[variant] and 
                        output_type in variant_means[variant][sentiment_model]):
                        variant_coverage += 1
            report_lines.append(f"  {variant:<15}: {variant_coverage}/{len(SENTIMENT_MODELS) * len(OUTPUT_TYPES)} combinations")
    
    report_lines.append("")
    report_lines.append("Analysis complete!")
    
    return "\n".join(report_lines)

def create_comparison_visualizations(variant_means):
    """
    Create comprehensive visualizations comparing the three LSTM variants.
    """
    # Create output directory
    viz_dir = "reports/output/feature_importance_analysis"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Performance comparison heatmaps for each output type
    for output_type in OUTPUT_TYPES:
        # Get all available metrics for this output type
        all_metrics = set()
        for variant in LSTM_VARIANTS:
            if variant in variant_means:
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in variant_means[variant] and 
                        output_type in variant_means[variant][sentiment_model]):
                        metrics = variant_means[variant][sentiment_model][output_type]['metrics']
                        all_metrics.update(metrics.keys())
        
        if not all_metrics:
            continue
        
        all_metrics = sorted(list(all_metrics))
        
        # Create subplots for each metric
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(all_metrics):
            ax = axes[idx]
            
            # Prepare data for heatmap
            heatmap_data = []
            for variant in LSTM_VARIANTS:
                row = []
                for sentiment_model in SENTIMENT_MODELS:
                    if (variant in variant_means and 
                        sentiment_model in variant_means[variant] and
                        output_type in variant_means[variant][sentiment_model] and
                        metric in variant_means[variant][sentiment_model][output_type]['metrics']):
                        
                        value = variant_means[variant][sentiment_model][output_type]['metrics'][metric]
                        row.append(value)
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
            
            if heatmap_data:
                heatmap_data = np.array(heatmap_data)
                
                # Create heatmap
                im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
                
                # Set labels
                ax.set_xticks(range(len(SENTIMENT_MODELS)))
                ax.set_xticklabels(SENTIMENT_MODELS, rotation=45, ha='right')
                ax.set_yticks(range(len(LSTM_VARIANTS)))
                ax.set_yticklabels(LSTM_VARIANTS)
                
                # Add value annotations
                for i in range(len(LSTM_VARIANTS)):
                    for j in range(len(SENTIMENT_MODELS)):
                        if not np.isnan(heatmap_data[i, j]):
                            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                         ha="center", va="center", color="white", fontsize=8, fontweight='bold')
                
                ax.set_title(f'{metric} - {output_type}')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric} - {output_type}')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        save_path = os.path.join(viz_dir, f"heatmap_{output_type}_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved: {save_path}")
    
    # 2. Overall performance summary - Individual metrics
    for output_type in OUTPUT_TYPES:
        # Get metrics for this output type
        if output_type == 'Binary_Price':
            metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        else:
            metrics = ['MAE', 'RMSE', 'MAPE', 'RSE', 'CORR']
            if output_type in ['Factor_Price', 'Float_Price']:
                metrics.append('MSLE')
        
        # Create subplot for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for j, metric in enumerate(metrics):
            ax = axes[j]
            
            # Collect data for this metric across all variants
            variant_values = []
            variant_names = []
            
            for variant in LSTM_VARIANTS:
                if variant not in variant_means:
                    continue
                    
                values = []
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in variant_means[variant] and
                        output_type in variant_means[variant][sentiment_model] and
                        'metrics' in variant_means[variant][sentiment_model][output_type] and
                        metric in variant_means[variant][sentiment_model][output_type]['metrics']):
                        
                        value = variant_means[variant][sentiment_model][output_type]['metrics'][metric]
                        if value is not None and not (isinstance(value, float) and np.isnan(value)):
                            values.append(value)
                
                if values:
                    variant_values.append(np.mean(values))
                    variant_names.append(variant)
            
            if variant_values:
                # Create bar plot for this metric
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
                bars = ax.bar(variant_names, variant_values, 
                             color=colors[:len(variant_names)], alpha=0.8)
                
                # Add value labels on bars
                for bar, value in zip(bars, variant_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
                
                ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Highlight best performing variant
                if metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CORR']:
                    # Higher is better
                    best_idx = np.argmax(variant_values)
                else:
                    # Lower is better
                    best_idx = np.argmin(variant_values)
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('red')
                bars[best_idx].set_linewidth(2)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric} - No Data')
        
        # Hide extra subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{output_type} - Performance by Metric and Variant', fontsize=16, fontweight='bold')
        plt.tight_layout()
        summary_path = os.path.join(viz_dir, f"performance_by_metric_{output_type}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance by metric plot saved: {summary_path}")
    
    # 3. Create comprehensive comparison plots
    create_comprehensive_comparison_plots(variant_means, viz_dir)
    
    # 4. Create feature importance analysis plots
    create_feature_importance_plots(variant_means, viz_dir)

def create_comprehensive_comparison_plots(variant_means, viz_dir):
    """
    Create comprehensive comparison plots showing detailed performance analysis.
    """
    # 1. Bar plot comparing variants across all metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, output_type in enumerate(OUTPUT_TYPES):
        ax = axes[i]
        
        # Get key metrics for this output type
        if output_type == 'Binary_Price':
            key_metrics = ['AUC', 'Accuracy', 'F1-Score']
        else:
            key_metrics = ['RMSE', 'MAE', 'CORR']
        
        # Prepare data
        x = np.arange(len(key_metrics))
        width = 0.25
        
        for j, variant in enumerate(LSTM_VARIANTS):
            if variant not in variant_means:
                continue
                
            values = []
            for metric in key_metrics:
                # Calculate average across sentiment models
                metric_values = []
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in variant_means[variant] and
                        output_type in variant_means[variant][sentiment_model] and
                        metric in variant_means[variant][sentiment_model][output_type]['metrics']):
                        
                        value = variant_means[variant][sentiment_model][output_type]['metrics'][metric]
                        if value is not None:
                            metric_values.append(value)
                
                if metric_values:
                    values.append(np.mean(metric_values))
                else:
                    values.append(0)
            
            if values:
                ax.bar(x + j * width, values, width, label=variant, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Average Values')
        ax.set_title(f'{output_type} - Average Performance by Variant')
        ax.set_xticks(x + width)
        ax.set_xticklabels(key_metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(viz_dir, "comprehensive_comparison_barplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive comparison plot saved: {save_path}")
    

def create_feature_importance_plots(variant_means, viz_dir):
    """
    Create plots specifically analyzing feature importance.
    """
    # 1. Feature count vs performance analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, output_type in enumerate(OUTPUT_TYPES):
        ax = axes[i]
        
        # Get key metric for this output type
        if output_type == 'Binary_Price':
            key_metric = 'AUC'
        else:
            key_metric = 'RMSE'
        
        # Collect data
        feature_counts = []
        performances = []
        variant_labels = []
        
        for variant in LSTM_VARIANTS:
            if variant not in variant_means:
                continue
                
            # Get feature count
            sample_features = None
            for sentiment_model in SENTIMENT_MODELS:
                for ot in OUTPUT_TYPES:
                    if (sentiment_model in variant_means[variant] and 
                        ot in variant_means[variant][sentiment_model]):
                        sample_features = variant_means[variant][sentiment_model][ot]['features']
                        break
                if sample_features:
                    break
            
            if sample_features:
                feature_count = len(sample_features)
                
                # Calculate average performance
                metric_values = []
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in variant_means[variant] and
                        output_type in variant_means[variant][sentiment_model] and
                        key_metric in variant_means[variant][sentiment_model][output_type]['metrics']):
                        
                        value = variant_means[variant][sentiment_model][output_type]['metrics'][key_metric]
                        if value is not None:
                            metric_values.append(value)
                
                if metric_values:
                    avg_performance = np.mean(metric_values)
                    feature_counts.append(feature_count)
                    performances.append(avg_performance)
                    variant_labels.append(variant)
        
        if feature_counts:
            # Create scatter plot
            scatter = ax.scatter(feature_counts, performances, s=200, alpha=0.7)
            
            # Add labels
            for j, label in enumerate(variant_labels):
                ax.annotate(label, (feature_counts[j], performances[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Number of Features')
            ax.set_ylabel(f'Average {key_metric}')
            ax.set_title(f'{output_type} - Feature Count vs Performance')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{output_type} - No Data')
    
    plt.tight_layout()
    save_path = os.path.join(viz_dir, "feature_count_vs_performance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved: {save_path}")

def main():
    """
    Main analysis function.
    """
    print("LSTM Feature Importance Analysis")
    print("=" * 50)
    print()
    
    # Collect all results
    print("Collecting results from all LSTM variants...")
    all_results = collect_all_results()
    
    if not all_results:
        print("No results found!")
        return
    
    # Calculate variant means
    print("Calculating mean performance across variants...")
    variant_means = calculate_variant_means(all_results)
    
    # Generate report
    print("Generating comparison report...")
    report = generate_comparison_report(variant_means)
    
    # Save report
    report_file = "reports/output/lstm_feature_importance_analysis.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {report_file}")
    
    # Create visualizations
    print("Creating comparison visualizations...")
    create_comparison_visualizations(variant_means)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print()
    
    # Print key findings
    print("Key Findings:")
    print("-" * 20)
    
    for output_type in OUTPUT_TYPES:
        print(f"\n{output_type}:")
        
        # Find best performing variant for key metrics
        key_metrics = ['AUC', 'Accuracy', 'F1-Score'] if output_type == 'Binary_Price' else ['RMSE', 'MAE', 'CORR']
        
        for metric in key_metrics:
            best_variant = None
            best_value = None
            best_sentiment = None
            
            for variant in LSTM_VARIANTS:
                if variant not in variant_means:
                    continue
                    
                for sentiment_model in SENTIMENT_MODELS:
                    if (sentiment_model in variant_means[variant] and 
                        output_type in variant_means[variant][sentiment_model] and
                        metric in variant_means[variant][sentiment_model][output_type]['metrics']):
                        
                        value = variant_means[variant][sentiment_model][output_type]['metrics'][metric]
                        if value is not None:
                            if best_value is None:
                                best_variant = variant
                                best_value = value
                                best_sentiment = sentiment_model
                            else:
                                # For classification metrics, higher is better
                                if metric in ['AUC', 'Accuracy', 'F1-Score', 'CORR']:
                                    if value > best_value:
                                        best_variant = variant
                                        best_value = value
                                        best_sentiment = sentiment_model
                                # For regression metrics, lower is better
                                else:
                                    if value < best_value:
                                        best_variant = variant
                                        best_value = value
                                        best_sentiment = sentiment_model
            
            if best_variant:
                print(f"  {metric}: {best_variant} ({best_sentiment}) = {best_value:.4f}")
    
    print("\nAnalysis complete!")
    print(f"Check the report at: {report_file}")
    print("Check visualizations at: reports/output/feature_importance_analysis/")

if __name__ == "__main__":
    main()
