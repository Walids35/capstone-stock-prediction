#!/usr/bin/env python3
"""
Script to generate reports showing mean metrics by architecture and prediction type.
This script reads the comprehensive analysis results and displays separate tables
for each prediction type (Binary_Price, Delta_Price, Factor_Price, Float_Price)
with mean values for each metric across all architectures.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

def load_results(file_path: str) -> Dict[str, Any]:
    """Load the comprehensive analysis results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics_by_architecture_and_type(data: Dict[str, Any], sentiment_model: str = 'roberta') -> Dict[str, Dict[str, List[float]]]:
    """
    Extract all metrics for each architecture organized by prediction type.
    Only includes results from the specified sentiment model.
    
    Returns:
        Dictionary with structure: {prediction_type: {metric_name: [values_for_each_architecture]}}
    """
    results_by_type = {}
    architectures = list(data.keys())
    
    # Get all prediction types
    prediction_types = set()
    for arch in architectures:
        if sentiment_model in data[arch]:
            for pred_type in data[arch][sentiment_model]:
                prediction_types.add(pred_type)
    
    # Process each prediction type separately
    for pred_type in prediction_types:
        metrics_by_arch = {}
        all_metrics = set()
        
        # First pass: collect all possible metric names for this prediction type
        for arch in architectures:
            if sentiment_model in data[arch] and pred_type in data[arch][sentiment_model]:
                for stock in data[arch][sentiment_model][pred_type]:
                    for metric in data[arch][sentiment_model][pred_type][stock]:
                        all_metrics.add(metric)
        
        # Initialize metrics dictionary for this prediction type
        for metric in all_metrics:
            metrics_by_arch[metric] = []
        
        # Second pass: collect values for each architecture for this prediction type
        for arch in architectures:
            arch_metrics = {metric: [] for metric in all_metrics}
            
            if sentiment_model in data[arch] and pred_type in data[arch][sentiment_model]:
                for stock in data[arch][sentiment_model][pred_type]:
                    metrics = data[arch][sentiment_model][pred_type][stock]
                    for metric, value in metrics.items():
                        if value != float('inf') and not pd.isna(value):
                            arch_metrics[metric].append(value)
            
            # Calculate mean for each metric for this architecture
            for metric in all_metrics:
                if arch_metrics[metric]:  # If there are values for this metric
                    mean_value = np.mean(arch_metrics[metric])
                    metrics_by_arch[metric].append(mean_value)
                else:
                    metrics_by_arch[metric].append(np.nan)
        
        results_by_type[pred_type] = (metrics_by_arch, architectures)
    
    return results_by_type

def create_report_dataframe(metrics_by_arch: Dict[str, List[float]], 
                          architectures: List[str]) -> pd.DataFrame:
    """Create a pandas DataFrame for the report."""
    
    # Create DataFrame
    df = pd.DataFrame(metrics_by_arch, index=architectures)
    
    # Transpose so metrics are rows and architectures are columns
    df = df.T
    
    # Sort metrics for better organization
    classification_metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    regression_metrics = ['MAE', 'RMSE', 'MAPE', 'RSE', 'CORR', 'MSLE']
    
    # Separate and order metrics
    available_class_metrics = [m for m in classification_metrics if m in df.index]
    available_reg_metrics = [m for m in regression_metrics if m in df.index]
    other_metrics = [m for m in df.index if m not in classification_metrics + regression_metrics]
    
    ordered_metrics = available_class_metrics + available_reg_metrics + other_metrics
    df = df.reindex(ordered_metrics)
    
    return df

def format_table(df: pd.DataFrame) -> str:
    """Format the DataFrame as a nice table string."""
    
    # Create a copy for formatting
    df_formatted = df.copy()
    
    # Format numbers to 4 decimal places
    for col in df_formatted.columns:
        df_formatted[col] = df_formatted[col].apply(
            lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
        )
    
    # Convert to string with nice formatting
    table_str = df_formatted.to_string(
        float_format=lambda x: f"{x:.4f}",
        na_rep="N/A",
        justify="center"
    )
    
    return table_str

def main():
    """Main function to generate the architecture metrics report by prediction type."""
    
    # Define file paths
    script_dir = Path(__file__).parent
    results_file = script_dir.parent / "reports" / "output" / "comprehensive_analysis_results.json"
    
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        return
    
    print("Loading comprehensive analysis results...")
    data = load_results(str(results_file))
    
    sentiment_model = 'roberta'  # Can be changed to 'finbert', 'deberta', etc.
    
    print(f"Extracting metrics by architecture and prediction type for {sentiment_model}...")
    results_by_type = extract_metrics_by_architecture_and_type(data, sentiment_model)
    
    # Generate separate reports for each prediction type
    prediction_type_names = {
        'Binary_Price': 'Binary Price Prediction',
        'Delta_Price': 'Delta Price Prediction', 
        'Factor_Price': 'Factor Price Prediction',
        'Float_Price': 'Float Price Prediction'
    }
    
    for pred_type, (metrics_by_arch, architectures) in results_by_type.items():
        print("\n" + "="*100)
        print(f"ARCHITECTURE PERFORMANCE REPORT - {prediction_type_names.get(pred_type, pred_type).upper()}")
        print(f"Sentiment Model: {sentiment_model.upper()}")
        print("="*100)
        
        print(f"\nThis table shows the mean value of each metric for {prediction_type_names.get(pred_type, pred_type)}:")
        print(f"- Sentiment model: {sentiment_model} only")
        print("- Stock symbols (AAPL, AMZN, MSFT, NFLX, TSLA)")
        
        # Create and display DataFrame for this prediction type
        df = create_report_dataframe(metrics_by_arch, architectures)
        print("\n" + format_table(df))
        
        # Print summary statistics for this prediction type
        print(f"\nSUMMARY STATISTICS - {prediction_type_names.get(pred_type, pred_type)}")
        print("-" * 60)
        print(f"Number of Architectures: {len(df.columns)}")
        print(f"Number of Metrics: {len(df.index)}")
        
        # Best performing architecture per metric for this prediction type
        print(f"\nBest Performing Architecture per Metric:")
        print("-" * 50)
        
        for metric in df.index:
            valid_values = df.loc[metric].dropna()
            if not valid_values.empty:
                if metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CORR']:
                    # Higher is better
                    best_arch = valid_values.idxmax()
                    best_value = valid_values.max()
                    print(f"{metric:12}: {best_arch:12} ({best_value:.4f})")
                elif metric in ['MAE', 'RMSE', 'MAPE', 'RSE', 'MSLE']:
                    # Lower is better
                    best_arch = valid_values.idxmin()
                    best_value = valid_values.min()
                    print(f"{metric:12}: {best_arch:12} ({best_value:.4f})")
        
        # Save to CSV for this prediction type
        safe_pred_type = pred_type.replace('_', '_').lower()
        output_file = script_dir.parent / "reports" / "output" / f"architecture_metrics_{safe_pred_type}_{sentiment_model}.csv"
        df.to_csv(output_file)
        print(f"\nReport saved to: {output_file}")
    
    # Print overall metric descriptions at the end
    print("\n" + "="*100)
    print("METRIC DESCRIPTIONS")
    print("="*100)
    print("Classification Metrics (higher is better):")
    print("  AUC: Area Under the ROC Curve")
    print("  Accuracy: Correct predictions / Total predictions")
    print("  Precision: True Positives / (True Positives + False Positives)")
    print("  Recall: True Positives / (True Positives + False Negatives)")
    print("  F1-Score: Harmonic mean of Precision and Recall")
    print("\nRegression Metrics:")
    print("  MAE: Mean Absolute Error (lower is better)")
    print("  RMSE: Root Mean Square Error (lower is better)")
    print("  MAPE: Mean Absolute Percentage Error (lower is better)")
    print("  RSE: Relative Squared Error (lower is better)")
    print("  CORR: Correlation coefficient (higher is better)")
    print("  MSLE: Mean Squared Log Error (lower is better)")

if __name__ == "__main__":
    main()
