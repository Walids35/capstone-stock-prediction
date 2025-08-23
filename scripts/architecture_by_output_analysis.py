#!/usr/bin/env python3
"""
Architecture by Output Type Analysis Script
Creates a table showing mean performance of each architecture (lstm, patchtst, tpatchgnn, timesnet) 
for each output type (Binary_Price, Delta_Price, Factor_Price, Float_Price).
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ArchitectureOutputAnalyzer:
    def __init__(self, simulation_dir=None, comprehensive_results_file=None):
        self.simulation_dir = simulation_dir
        self.comprehensive_results_file = comprehensive_results_file
        self.trading_data = {}
        self.comprehensive_data = {}
        
    def load_trading_simulation_data(self):
        """Load trading simulation data (currently only Binary_Price available)."""
        if not self.simulation_dir or not os.path.exists(self.simulation_dir):
            print("Trading simulation directory not found")
            return
            
        print("Loading trading simulation data...")
        
        json_files = [f for f in os.listdir(self.simulation_dir) 
                     if f.endswith('_trading_analysis.json')]
        
        for filename in json_files:
            file_path = os.path.join(self.simulation_dir, filename)
            
            # Parse filename to extract components
            parts = filename.replace('_trading_analysis.json', '').split('_')
            if len(parts) >= 4:
                ticker = parts[0]
                architecture = parts[1]
                prediction_type = parts[2]  # This should be Binary_Price for trading data
                sentiment_model = '_'.join(parts[3:])
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Store data with metadata
                    if architecture not in self.trading_data:
                        self.trading_data[architecture] = {}
                    if prediction_type not in self.trading_data[architecture]:
                        self.trading_data[architecture][prediction_type] = []
                    
                    # Extract relevant metrics
                    self.trading_data[architecture][prediction_type].append({
                        'ticker': ticker,
                        'sentiment_model': sentiment_model,
                        'profit_percentage': data.get('profit_percentage', 0),
                        'total_profit': data.get('total_profit', 0),
                        'buy_count': data.get('buy_count', 0),
                        'sell_count': data.get('sell_count', 0),
                        'total_trades': data.get('buy_count', 0) + data.get('sell_count', 0),
                        'final_balance': data.get('final_balance', 0)
                    })
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Loaded trading data for {len(self.trading_data)} architectures")
    
    def load_comprehensive_analysis_data(self):
        """Load comprehensive analysis data (includes all output types)."""
        if not self.comprehensive_results_file or not os.path.exists(self.comprehensive_results_file):
            print("Comprehensive results file not found")
            return
            
        print("Loading comprehensive analysis data...")
        
        try:
            with open(self.comprehensive_results_file, 'r') as f:
                data = json.load(f)
            
            # Reorganize data by architecture and output type
            for model_arch, arch_data in data.items():
                arch_name = model_arch.lower()  # lstm, patchtst, timesnet, tpatchgnn
                
                if arch_name not in self.comprehensive_data:
                    self.comprehensive_data[arch_name] = {}
                
                for sentiment_model, sentiment_data in arch_data.items():
                    for output_type, output_data in sentiment_data.items():
                        if output_type not in self.comprehensive_data[arch_name]:
                            self.comprehensive_data[arch_name][output_type] = []
                        
                        for ticker, metrics in output_data.items():
                            self.comprehensive_data[arch_name][output_type].append({
                                'ticker': ticker,
                                'sentiment_model': sentiment_model,
                                'metrics': metrics
                            })
                            
        except Exception as e:
            print(f"Error loading comprehensive results: {e}")
        
        print(f"Loaded comprehensive data for {len(self.comprehensive_data)} architectures")
    
    def create_trading_summary_table(self):
        """Create summary table for trading simulation data (Binary_Price only)."""
        if not self.trading_data:
            print("No trading data available")
            return None
            
        print("\n" + "="*80)
        print("TRADING SIMULATION ANALYSIS BY ARCHITECTURE AND OUTPUT TYPE")
        print("="*80)
        
        summary_data = []
        
        for architecture in sorted(self.trading_data.keys()):
            for output_type in sorted(self.trading_data[architecture].keys()):
                records = self.trading_data[architecture][output_type]
                
                if records:
                    # Calculate means
                    profit_percentages = [r['profit_percentage'] for r in records]
                    total_profits = [r['total_profit'] for r in records]
                    trade_counts = [r['total_trades'] for r in records]
                    final_balances = [r['final_balance'] for r in records]
                    
                    summary_data.append({
                        'Architecture': architecture,
                        'Output_Type': output_type,
                        'Count': len(records),
                        'Mean_Profit_Percentage': np.mean(profit_percentages),
                        'Std_Profit_Percentage': np.std(profit_percentages),
                        'Mean_Total_Profit': np.mean(total_profits),
                        'Mean_Trade_Count': np.mean(trade_counts),
                        'Mean_Final_Balance': np.mean(final_balances),
                        'Profitable_Models': sum(1 for p in profit_percentages if p > 0),
                        'Success_Rate': sum(1 for p in profit_percentages if p > 0) / len(profit_percentages) * 100
                    })
        
        df = pd.DataFrame(summary_data)
        
        if len(df) > 0:
            # Format the display
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', '{:.4f}'.format)
            
            print("\nTrading Performance by Architecture and Output Type:")
            print("-" * 80)
            print(df.to_string(index=False))
            
            # Create a pivot table for easier comparison
            pivot_profit = df.pivot(index='Architecture', columns='Output_Type', values='Mean_Profit_Percentage')
            print("\n\nPivot Table - Mean Profit Percentage by Architecture and Output Type:")
            print("-" * 80)
            print(pivot_profit.to_string())
            
            return df
        else:
            print("No data to display")
            return None
    
    def create_comprehensive_summary_table(self):
        """Create summary table for comprehensive analysis data (all output types)."""
        if not self.comprehensive_data:
            print("No comprehensive data available")
            return None
            
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS BY ARCHITECTURE AND OUTPUT TYPE")
        print("="*80)
        
        summary_data = []
        
        # Metrics to analyze (common across output types)
        target_metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'RSE', 'CORR', 'MAE', 'RMSE']
        
        for architecture in sorted(self.comprehensive_data.keys()):
            for output_type in sorted(self.comprehensive_data[architecture].keys()):
                records = self.comprehensive_data[architecture][output_type]
                
                if records:
                    # Initialize metric storage
                    metric_values = defaultdict(list)
                    
                    # Collect all metric values
                    for record in records:
                        metrics = record['metrics']
                        for metric_name, value in metrics.items():
                            if metric_name in target_metrics and isinstance(value, (int, float)) and not np.isnan(value):
                                metric_values[metric_name].append(value)
                    
                    # Calculate means for each metric
                    row_data = {
                        'Architecture': architecture,
                        'Output_Type': output_type,
                        'Model_Count': len(records)
                    }
                    
                    for metric in target_metrics:
                        if metric in metric_values and len(metric_values[metric]) > 0:
                            row_data[f'Mean_{metric}'] = np.mean(metric_values[metric])
                            row_data[f'Std_{metric}'] = np.std(metric_values[metric])
                            row_data[f'Count_{metric}'] = len(metric_values[metric])
                        else:
                            row_data[f'Mean_{metric}'] = np.nan
                            row_data[f'Std_{metric}'] = np.nan
                            row_data[f'Count_{metric}'] = 0
                    
                    summary_data.append(row_data)
        
        df = pd.DataFrame(summary_data)
        
        if len(df) > 0:
            # Display main metrics
            main_metrics = ['Mean_AUC', 'Mean_Accuracy', 'Mean_F1-Score', 'Mean_RSE', 'Mean_CORR', 'Mean_MAE']
            display_cols = ['Architecture', 'Output_Type', 'Model_Count'] + main_metrics
            
            # Filter columns that exist
            existing_cols = [col for col in display_cols if col in df.columns]
            
            print("\nComprehensive Performance by Architecture and Output Type:")
            print("-" * 80)
            print(df[existing_cols].round(4).to_string(index=False))
            
            # Create pivot tables for main metrics
            for metric in ['Mean_AUC', 'Mean_Accuracy', 'Mean_F1-Score']:
                if metric in df.columns:
                    pivot_table = df.pivot(index='Architecture', columns='Output_Type', values=metric)
                    print(f"\n\nPivot Table - {metric} by Architecture and Output Type:")
                    print("-" * 60)
                    print(pivot_table.round(4).to_string())
            
            return df
        else:
            print("No comprehensive data to display")
            return None
    
    def save_results_to_csv(self, trading_df, comprehensive_df, output_dir):
        """Save both trading and comprehensive results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        if trading_df is not None:
            trading_csv_path = os.path.join(output_dir, 'architecture_output_trading_analysis.csv')
            trading_df.to_csv(trading_csv_path, index=False)
            print(f"\nTrading analysis saved to: {trading_csv_path}")
            
            # Also save pivot table
            pivot_profit = trading_df.pivot(index='Architecture', columns='Output_Type', values='Mean_Profit_Percentage')
            pivot_csv_path = os.path.join(output_dir, 'architecture_output_trading_pivot.csv')
            pivot_profit.to_csv(pivot_csv_path)
            print(f"Trading pivot table saved to: {pivot_csv_path}")
        
        if comprehensive_df is not None:
            comprehensive_csv_path = os.path.join(output_dir, 'architecture_output_comprehensive_analysis.csv')
            comprehensive_df.to_csv(comprehensive_csv_path, index=False)
            print(f"Comprehensive analysis saved to: {comprehensive_csv_path}")
            
            # Save pivot tables for main metrics
            main_metrics = ['Mean_AUC', 'Mean_Accuracy', 'Mean_F1-Score', 'Mean_RSE', 'Mean_CORR', 'Mean_MAE']
            for metric in main_metrics:
                if metric in comprehensive_df.columns:
                    pivot_table = comprehensive_df.pivot(index='Architecture', columns='Output_Type', values=metric)
                    pivot_csv_path = os.path.join(output_dir, f'architecture_output_{metric}_pivot.csv')
                    pivot_table.to_csv(pivot_csv_path)
                    print(f"{metric} pivot table saved to: {pivot_csv_path}")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("="*80)
        print("ARCHITECTURE BY OUTPUT TYPE ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_trading_simulation_data()
        self.load_comprehensive_analysis_data()
        
        # Create analyses
        trading_df = self.create_trading_summary_table()
        comprehensive_df = self.create_comprehensive_summary_table()
        
        # Save results
        output_dir = os.path.join(os.path.dirname(self.simulation_dir or '.'), 'output', 'architecture_output_analysis')
        self.save_results_to_csv(trading_df, comprehensive_df, output_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return trading_df, comprehensive_df

def main():
    """Main execution function."""
    # Define paths
    script_dir = Path(__file__).parent
    reports_dir = script_dir.parent / "reports"
    simulation_dir = reports_dir / "simulation"
    comprehensive_file = reports_dir / "output" / "comprehensive_analysis_results.json"
    
    # Create analyzer and run analysis
    analyzer = ArchitectureOutputAnalyzer(
        simulation_dir=str(simulation_dir) if simulation_dir.exists() else None,
        comprehensive_results_file=str(comprehensive_file) if comprehensive_file.exists() else None
    )
    
    trading_df, comprehensive_df = analyzer.run_analysis()
    
    return trading_df, comprehensive_df

if __name__ == "__main__":
    main()
