#!/usr/bin/env python3
"""
Simple Architecture by Output Type Table Generator
Creates a clean table showing mean performance metrics for each model architecture 
across all output types (Binary_Price, Delta_Price, Factor_Price, Float_Price).
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def create_architecture_output_table():
    """Create a comprehensive table of architecture performance by output type."""
    
    # Define paths
    script_dir = Path(__file__).parent
    reports_dir = script_dir.parent / "reports"
    comprehensive_file = reports_dir / "output" / "comprehensive_analysis_results.json"
    
    if not comprehensive_file.exists():
        print(f"Comprehensive results file not found: {comprehensive_file}")
        return None
    
    print("Loading comprehensive analysis data...")
    
    # Load comprehensive analysis data
    with open(comprehensive_file, 'r') as f:
        data = json.load(f)
    
    # Initialize results storage
    results = defaultdict(lambda: defaultdict(list))
    
    # Process data
    for model_arch, arch_data in data.items():
        arch_name = model_arch.lower()  # Convert to lowercase for consistency
        
        for sentiment_model, sentiment_data in arch_data.items():
            for output_type, output_data in sentiment_data.items():
                for ticker, metrics in output_data.items():
                    # Store metrics for this architecture-output_type combination
                    results[arch_name][output_type].append(metrics)
    
    # Create summary table
    table_data = []
    
    # Define the metrics we want to calculate means for
    metric_sets = {
        'Binary_Price': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Delta_Price': ['RSE', 'CORR', 'MAE', 'RMSE', 'MSLE'],
        'Factor_Price': ['RSE', 'CORR', 'MAE', 'RMSE', 'MSLE'],
        'Float_Price': ['RSE', 'CORR', 'MAE', 'RMSE', 'MSLE']
    }
    
    # Process each architecture
    for architecture in ['lstm', 'patchtst', 'timesnet', 'tpatchgnn']:
        row = {'Architecture': architecture}
        
        for output_type in ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']:
            if output_type in results[architecture]:
                records = results[architecture][output_type]
                
                # Calculate means for relevant metrics
                relevant_metrics = metric_sets[output_type]
                metric_values = defaultdict(list)
                
                for record in records:
                    for metric in relevant_metrics:
                        if metric in record and isinstance(record[metric], (int, float)) and not np.isnan(record[metric]):
                            metric_values[metric].append(record[metric])
                
                # Store the primary metric for each output type
                if output_type == 'Binary_Price':
                    primary_metric = 'Accuracy'
                else:
                    primary_metric = 'CORR'  # Use correlation as primary for regression tasks
                
                if primary_metric in metric_values and len(metric_values[primary_metric]) > 0:
                    row[f'{output_type}_Mean'] = np.mean(metric_values[primary_metric])
                    row[f'{output_type}_Count'] = len(metric_values[primary_metric])
                else:
                    row[f'{output_type}_Mean'] = np.nan
                    row[f'{output_type}_Count'] = 0
            else:
                row[f'{output_type}_Mean'] = np.nan
                row[f'{output_type}_Count'] = 0
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display the table
    print("\n" + "="*100)
    print("ARCHITECTURE PERFORMANCE BY OUTPUT TYPE")
    print("="*100)
    print("Primary Metrics: Binary_Price=Accuracy, Others=Correlation")
    print("-"*100)
    
    # Format for better display
    display_df = df.copy()
    
    # Round numeric columns
    numeric_cols = [col for col in display_df.columns if col.endswith('_Mean')]
    for col in numeric_cols:
        display_df[col] = display_df[col].round(4)
    
    print(display_df.to_string(index=False))
    
    # Create a cleaner pivot table
    print("\n" + "="*80)
    print("PIVOT TABLE - PRIMARY METRICS BY ARCHITECTURE AND OUTPUT TYPE")
    print("="*80)
    
    # Reshape for pivot table
    pivot_data = []
    for _, row in df.iterrows():
        for output_type in ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']:
            pivot_data.append({
                'Architecture': row['Architecture'],
                'Output_Type': output_type,
                'Mean_Performance': row[f'{output_type}_Mean'],
                'Count': row[f'{output_type}_Count']
            })
    
    pivot_df = pd.DataFrame(pivot_data)
    pivot_table = pivot_df.pivot(index='Architecture', columns='Output_Type', values='Mean_Performance')
    
    print(pivot_table.round(4).to_string())
    
    # Save to CSV
    output_dir = reports_dir / "output" / "architecture_output_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed table
    csv_path = output_dir / "architecture_output_mean_performance_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed table saved to: {csv_path}")
    
    # Save pivot table
    pivot_csv_path = output_dir / "architecture_output_pivot_table.csv"
    pivot_table.to_csv(pivot_csv_path)
    print(f"Pivot table saved to: {pivot_csv_path}")
    
    # Create a comprehensive metrics table
    create_comprehensive_metrics_table(results, output_dir)
    
    return df, pivot_table

def create_comprehensive_metrics_table(results, output_dir):
    """Create a more comprehensive table with multiple metrics per output type."""
    
    print("\n" + "="*120)
    print("COMPREHENSIVE METRICS TABLE BY ARCHITECTURE AND OUTPUT TYPE")
    print("="*120)
    
    # Define metrics for each output type
    metric_sets = {
        'Binary_Price': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Delta_Price': ['RSE', 'CORR', 'MAE', 'RMSE'],
        'Factor_Price': ['RSE', 'CORR', 'MAE', 'RMSE'],
        'Float_Price': ['RSE', 'CORR', 'MAE', 'RMSE']
    }
    
    comprehensive_data = []
    
    for architecture in ['lstm', 'patchtst', 'timesnet', 'tpatchgnn']:
        for output_type in ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']:
            if output_type in results[architecture]:
                records = results[architecture][output_type]
                relevant_metrics = metric_sets[output_type]
                
                row = {
                    'Architecture': architecture,
                    'Output_Type': output_type,
                    'Model_Count': len(records)
                }
                
                # Calculate means for all relevant metrics
                metric_values = defaultdict(list)
                
                for record in records:
                    for metric in relevant_metrics:
                        if metric in record and isinstance(record[metric], (int, float)) and not np.isnan(record[metric]):
                            metric_values[metric].append(record[metric])
                
                for metric in relevant_metrics:
                    if metric in metric_values and len(metric_values[metric]) > 0:
                        row[f'Mean_{metric}'] = np.mean(metric_values[metric])
                        row[f'Std_{metric}'] = np.std(metric_values[metric])
                    else:
                        row[f'Mean_{metric}'] = np.nan
                        row[f'Std_{metric}'] = np.nan
                
                comprehensive_data.append(row)
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    
    # Display by output type
    for output_type in ['Binary_Price', 'Delta_Price', 'Factor_Price', 'Float_Price']:
        print(f"\n{output_type} Results:")
        print("-" * 60)
        
        subset = comprehensive_df[comprehensive_df['Output_Type'] == output_type].copy()
        if len(subset) > 0:
            # Select main columns for display
            if output_type == 'Binary_Price':
                display_cols = ['Architecture', 'Model_Count', 'Mean_AUC', 'Mean_Accuracy', 'Mean_F1-Score']
            else:
                display_cols = ['Architecture', 'Model_Count', 'Mean_RSE', 'Mean_CORR', 'Mean_MAE']
            
            # Filter existing columns
            existing_cols = [col for col in display_cols if col in subset.columns]
            print(subset[existing_cols].round(4).to_string(index=False))
        else:
            print("No data available")
    
    # Save comprehensive table
    comprehensive_csv_path = output_dir / "architecture_output_comprehensive_metrics.csv"
    comprehensive_df.to_csv(comprehensive_csv_path, index=False)
    print(f"\nComprehensive metrics table saved to: {comprehensive_csv_path}")
    
    return comprehensive_df

def main():
    """Main execution function."""
    df, pivot_table = create_architecture_output_table()
    return df, pivot_table

if __name__ == "__main__":
    main()
