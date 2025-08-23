#!/usr/bin/env python3
"""
Visualization Generation Script
Creates charts and plots from the comprehensive analysis results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define the constants
TICKERS = ['AAPL', 'AMZN', 'MSFT', 'NFLX', 'TSLA']
SENTIMENT_MODELS = ['deberta', 'finbert', 'lr', 'rf', 'roberta', 'svm']
OUTPUT_TYPES = ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']
MODEL_TYPES = ['LSTM', 'PatchTST', 'TimesNet', 'tPatchGNN']

def load_comprehensive_results(json_file: str) -> dict:
    """
    Load the comprehensive analysis results from JSON file.
    """
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        print("Please run the comprehensive analysis first.")
        return {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_model_comparison_heatmap(data: dict, output_type: str, metric: str, save_path: str):
    """
    Create a heatmap comparing models across sentiment models and tickers.
    """
    # Prepare data for heatmap
    heatmap_data = []
    sentiment_models = []
    tickers = []
    
    for sentiment_model in SENTIMENT_MODELS:
        for ticker in TICKERS:
            values = []
            for model_type in MODEL_TYPES:
                if (model_type in data and 
                    sentiment_model in data[model_type] and
                    output_type in data[model_type][sentiment_model] and
                    ticker in data[model_type][sentiment_model][output_type] and
                    metric in data[model_type][sentiment_model][output_type][ticker]):
                    
                    value = data[model_type][sentiment_model][output_type][ticker][metric]
                    values.append(value)
                else:
                    values.append(np.nan)
            
            if any(not np.isnan(v) for v in values):
                heatmap_data.append(values)
                sentiment_models.append(sentiment_model)
                tickers.append(ticker)
    
    if not heatmap_data:
        print(f"No data available for {output_type} - {metric}")
        return
    
    heatmap_data = np.array(heatmap_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(MODEL_TYPES)))
    ax.set_xticklabels(MODEL_TYPES, rotation=45, ha='right')
    ax.set_yticks(range(len(sentiment_models)))
    ax.set_yticklabels([f"{s}_{t}" for s, t in zip(sentiment_models, tickers)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=20)
    
    # Add title
    plt.title(f'{metric} Comparison - {output_type}\n(Model Types vs Sentiment Models + Tickers)', 
              fontsize=14, pad=20)
    
    # Add value annotations
    for i in range(len(sentiment_models)):
        for j in range(len(MODEL_TYPES)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {save_path}")

def create_metric_comparison_barplot(data: dict, output_type: str, model_type: str, save_path: str):
    """
    Create a bar plot comparing metrics for a specific model type and output type.
    """
    # Collect all available metrics
    all_metrics = set()
    for sentiment_model in SENTIMENT_MODELS:
        for ticker in TICKERS:
            if (model_type in data and 
                sentiment_model in data[model_type] and
                output_type in data[model_type][sentiment_model] and
                ticker in data[model_type][sentiment_model][output_type]):
                
                metrics = data[model_type][sentiment_model][output_type][ticker]
                all_metrics.update(metrics.keys())
    
    if not all_metrics:
        print(f"No data available for {model_type} - {output_type}")
        return
    
    all_metrics = sorted(list(all_metrics))
    
    # Prepare data for plotting
    plot_data = []
    labels = []
    
    for sentiment_model in SENTIMENT_MODELS:
        for ticker in TICKERS:
            if (model_type in data and 
                sentiment_model in data[model_type] and
                output_type in data[model_type][sentiment_model] and
                ticker in data[model_type][sentiment_model][output_type]):
                
                metrics = data[model_type][sentiment_model][output_type][ticker]
                values = [metrics.get(metric, np.nan) for metric in all_metrics]
                
                if any(not np.isnan(v) for v in values):
                    plot_data.append(values)
                    labels.append(f"{sentiment_model}_{ticker}")
    
    if not plot_data:
        print(f"No valid data for {model_type} - {output_type}")
        return
    
    plot_data = np.array(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(all_metrics))
    width = 0.8 / len(plot_data)
    
    for i, (values, label) in enumerate(zip(plot_data, labels)):
        valid_indices = ~np.isnan(values)
        if any(valid_indices):
            ax.bar(x[valid_indices] + i * width, values[valid_indices], 
                   width, label=label, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(f'Metric Comparison - {model_type} - {output_type}')
    ax.set_xticks(x + width * (len(plot_data) - 1) / 2)
    ax.set_xticklabels(all_metrics, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved: {save_path}")

def create_model_performance_summary(data: dict, save_path: str):
    """
    Create a summary plot showing overall model performance.
    """
    # Calculate average performance for each model type
    model_performance = {}
    
    for model_type in MODEL_TYPES:
        if model_type not in data:
            continue
            
        model_performance[model_type] = {}
        
        for output_type in OUTPUT_TYPES:
            all_metrics = {}
            count = 0
            
            for sentiment_model in SENTIMENT_MODELS:
                for ticker in TICKERS:
                    if (sentiment_model in data[model_type] and
                        output_type in data[model_type][sentiment_model] and
                        ticker in data[model_type][sentiment_model][output_type]):
                        
                        metrics = data[model_type][sentiment_model][output_type][ticker]
                        for metric, value in metrics.items():
                            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                if metric not in all_metrics:
                                    all_metrics[metric] = []
                                all_metrics[metric].append(value)
                        count += 1
            
            # Calculate averages
            avg_metrics = {}
            for metric, values in all_metrics.items():
                if values:
                    avg_metrics[metric] = np.mean(values)
            
            if avg_metrics:
                model_performance[model_type][output_type] = avg_metrics
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, output_type in enumerate(OUTPUT_TYPES):
        ax = axes[i]
        
        # Collect data for this output type
        plot_data = []
        model_names = []
        
        for model_type in MODEL_TYPES:
            if (model_type in model_performance and 
                output_type in model_performance[model_type]):
                
                metrics = model_performance[model_type][output_type]
                if metrics:
                    plot_data.append(list(metrics.values()))
                    model_names.append(model_type)
        
        if plot_data:
            # Find common metrics across all models for this output type
            all_metrics = set()
            for model_type in model_names:
                if output_type in model_performance[model_type]:
                    all_metrics.update(model_performance[model_type][output_type].keys())
            
            if all_metrics:
                metric_names = sorted(list(all_metrics))
                
                # Create consistent plot data
                consistent_plot_data = []
                for model_type in model_names:
                    if output_type in model_performance[model_type]:
                        row = []
                        for metric in metric_names:
                            if metric in model_performance[model_type][output_type]:
                                row.append(model_performance[model_type][output_type][metric])
                            else:
                                row.append(np.nan)
                        consistent_plot_data.append(row)
                    else:
                        consistent_plot_data.append([np.nan] * len(metric_names))
                
                if consistent_plot_data:
                    plot_data = np.array(consistent_plot_data)
                    
                    # Create heatmap
                    im = ax.imshow(plot_data, cmap='RdYlBu_r', aspect='auto')
                    
                    # Set labels
                    ax.set_xticks(range(len(metric_names)))
                    ax.set_xticklabels(metric_names, rotation=45, ha='right')
                    ax.set_yticks(range(len(model_names)))
                    ax.set_yticklabels(model_names)
                    
                    # Add value annotations
                    for j in range(len(model_names)):
                        for k in range(len(metric_names)):
                            if not np.isnan(plot_data[j, k]):
                                text = ax.text(k, j, f'{plot_data[j, k]:.3f}',
                                             ha="center", va="center", color="black", fontsize=8)
                    
                    ax.set_title(f'{output_type} - Average Performance')
                else:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{output_type} - No Data')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{output_type} - No Data')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{output_type} - No Data')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved: {save_path}")

def main():
    """
    Main function to generate all visualizations.
    """
    print("Generating visualizations from comprehensive analysis...")
    print("=" * 60)
    
    # Load data
    json_file = "reports/output/comprehensive_analysis_results.json"
    data = load_comprehensive_results(json_file)
    
    if not data:
        return
    
    # Create output directory for visualizations
    viz_dir = "reports/output/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Creating visualizations...")
    
    # 1. Model comparison heatmaps for key metrics
    key_metrics = {
        'Binary_Price': ['AUC', 'Accuracy', 'F1-Score'],
        'Delta_Price': ['RMSE', 'MAE', 'RSE', 'CORR'],
        'Factor_Price': ['RMSE', 'MAE', 'RSE', 'CORR'],
        'Float_Price': ['RMSE', 'MAE', 'RSE', 'CORR']
    }
    
    for output_type, metrics in key_metrics.items():
        for metric in metrics:
            save_path = os.path.join(viz_dir, f"heatmap_{output_type}_{metric}.png")
            create_model_comparison_heatmap(data, output_type, metric, save_path)
    
    # 2. Metric comparison bar plots for each model type
    for model_type in MODEL_TYPES:
        for output_type in OUTPUT_TYPES:
            save_path = os.path.join(viz_dir, f"barplot_{model_type}_{output_type}.png")
            create_metric_comparison_barplot(data, output_type, model_type, save_path)
    
    # 3. Overall model performance summary
    summary_path = os.path.join(viz_dir, "model_performance_summary.png")
    create_model_performance_summary(data, summary_path)
    
    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print(f"All visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    
    # List generated files
    for file in os.listdir(viz_dir):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
