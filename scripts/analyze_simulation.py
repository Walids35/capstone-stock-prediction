#!/usr/bin/env python3
"""
Trading Simulation Analysis Script
Analyzes trading results from simulation data excluding specific financial metrics.
Focuses on: profit percentage, buy/sell counts, and trading performance.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TradingAnalyzer:
    def __init__(self, simulation_dir):
        self.simulation_dir = simulation_dir
        self.results = {}
        
    def load_simulation_data(self):
        """Load all trading simulation JSON files."""
        print("Loading simulation data...")
        
        json_files = [f for f in os.listdir(self.simulation_dir) 
                     if f.endswith('_trading_analysis.json')]
        
        for filename in json_files:
            file_path = os.path.join(self.simulation_dir, filename)
            
            # Parse filename to extract components
            parts = filename.replace('_trading_analysis.json', '').split('_')
            if len(parts) >= 4:
                ticker = parts[0]
                architecture = parts[1]
                prediction_type = parts[2]
                sentiment_model = '_'.join(parts[3:])  # Handle multi-word sentiment models
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Store data with metadata
                    key = f"{ticker}_{architecture}_{prediction_type}_{sentiment_model}"
                    self.results[key] = {
                        'ticker': ticker,
                        'architecture': architecture,
                        'prediction_type': prediction_type,
                        'sentiment_model': sentiment_model,
                        'data': data,
                        'filename': filename
                    }
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.results)} simulation results")
        return len(self.results)
    
    def create_performance_dataframe(self):
        """Create a comprehensive DataFrame with relevant metrics only."""
        records = []
        
        for key, result in self.results.items():
            data = result['data']
            
            # Extract relevant metrics (excluding the specified ones)
            record = {
                'Ticker': result['ticker'],
                'Architecture': result['architecture'],
                'Prediction_Type': result['prediction_type'],
                'Sentiment_Model': result['sentiment_model'],
                'Initial_Investment': data.get('initial_investment', 0),
                'Final_Balance': data.get('final_balance', 0),
                'Total_Profit': data.get('total_profit', 0),
                'Profit_Percentage': data.get('profit_percentage', 0),
                'Buy_Count': data.get('buy_count', 0),
                'Sell_Count': data.get('sell_count', 0),
                'Total_Trades': data.get('buy_count', 0) + data.get('sell_count', 0),
                'Key': key
            }
            
            # Calculate additional metrics
            if record['Total_Trades'] > 0:
                record['Avg_Profit_Per_Trade'] = record['Total_Profit'] / record['Total_Trades']
            else:
                record['Avg_Profit_Per_Trade'] = 0
                
            record['Trading_Activity'] = record['Total_Trades']
            record['Profitable'] = 1 if record['Total_Profit'] > 0 else 0
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_architecture_report(self, df):
        """Generate performance report by architecture."""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS BY ARCHITECTURE")
        print("="*80)
        
        arch_summary = df.groupby('Architecture').agg({
            'Profit_Percentage': ['mean', 'std', 'min', 'max', 'count'],
            'Total_Profit': ['mean', 'sum'],
            'Buy_Count': 'mean',
            'Sell_Count': 'mean',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        
        # Flatten column names
        arch_summary.columns = ['_'.join(col).strip() for col in arch_summary.columns]
        
        # Calculate success rate
        arch_summary['Success_Rate'] = (arch_summary['Profitable_sum'] / 
                                      arch_summary['Profit_Percentage_count'] * 100).round(2)
        
        print("\nArchitecture Performance Summary:")
        print("-" * 50)
        print(arch_summary.to_string())
        
        # Find best performing architecture
        best_arch = arch_summary['Profit_Percentage_mean'].idxmax()
        best_profit = arch_summary.loc[best_arch, 'Profit_Percentage_mean']
        
        print(f"\nBest Performing Architecture: {best_arch}")
        print(f"Average Profit Percentage: {best_profit:.4f}%")
        
        return arch_summary
    
    def generate_sentiment_model_report(self, df):
        """Generate performance report by sentiment model."""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS BY SENTIMENT MODEL")
        print("="*80)
        
        sent_summary = df.groupby('Sentiment_Model').agg({
            'Profit_Percentage': ['mean', 'std', 'min', 'max', 'count'],
            'Total_Profit': ['mean', 'sum'],
            'Buy_Count': 'mean',
            'Sell_Count': 'mean',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        
        # Flatten column names
        sent_summary.columns = ['_'.join(col).strip() for col in sent_summary.columns]
        
        # Calculate success rate
        sent_summary['Success_Rate'] = (sent_summary['Profitable_sum'] / 
                                      sent_summary['Profit_Percentage_count'] * 100).round(2)
        
        print("\nSentiment Model Performance Summary:")
        print("-" * 50)
        print(sent_summary.to_string())
        
        # Find best performing sentiment model
        best_sent = sent_summary['Profit_Percentage_mean'].idxmax()
        best_profit = sent_summary.loc[best_sent, 'Profit_Percentage_mean']
        
        print(f"\nBest Performing Sentiment Model: {best_sent}")
        print(f"Average Profit Percentage: {best_profit:.4f}%")
        
        return sent_summary
    
    def generate_ticker_report(self, df):
        """Generate performance report by ticker."""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS BY TICKER")
        print("="*80)
        
        ticker_summary = df.groupby('Ticker').agg({
            'Profit_Percentage': ['mean', 'std', 'min', 'max', 'count'],
            'Total_Profit': ['mean', 'sum'],
            'Buy_Count': 'mean',
            'Sell_Count': 'mean',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        
        # Flatten column names
        ticker_summary.columns = ['_'.join(col).strip() for col in ticker_summary.columns]
        
        # Calculate success rate
        ticker_summary['Success_Rate'] = (ticker_summary['Profitable_sum'] / 
                                        ticker_summary['Profit_Percentage_count'] * 100).round(2)
        
        print("\nTicker Performance Summary:")
        print("-" * 50)
        print(ticker_summary.to_string())
        
        # Find best performing ticker
        best_ticker = ticker_summary['Profit_Percentage_mean'].idxmax()
        best_profit = ticker_summary.loc[best_ticker, 'Profit_Percentage_mean']
        
        print(f"\nBest Performing Ticker: {best_ticker}")
        print(f"Average Profit Percentage: {best_profit:.4f}%")
        
        return ticker_summary
    
    def generate_model_combination_report(self, df):
        """Generate detailed report for each model combination."""
        print("\n" + "="*80)
        print("DETAILED MODEL COMBINATION ANALYSIS")
        print("="*80)
        
        # Group by architecture and sentiment model combination
        combination_summary = df.groupby(['Architecture', 'Sentiment_Model']).agg({
            'Profit_Percentage': ['mean', 'std', 'count'],
            'Total_Profit': 'sum',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        
        # Flatten column names
        combination_summary.columns = ['_'.join(col).strip() for col in combination_summary.columns]
        
        # Calculate success rate
        combination_summary['Success_Rate'] = (combination_summary['Profitable_sum'] / 
                                             combination_summary['Profit_Percentage_count'] * 100).round(2)
        
        # Sort by average profit percentage
        combination_summary = combination_summary.sort_values('Profit_Percentage_mean', ascending=False)
        
        print("\nTop 10 Model Combinations by Average Profit Percentage:")
        print("-" * 70)
        print(combination_summary.head(10).to_string())
        
        return combination_summary
    
    def generate_top_performers_report(self, df, top_n=10):
        """Generate report for top individual performers."""
        print("\n" + "="*80)
        print(f"TOP {top_n} INDIVIDUAL PERFORMERS")
        print("="*80)
        
        # Sort by profit percentage
        top_performers = df.nlargest(top_n, 'Profit_Percentage')[
            ['Ticker', 'Architecture', 'Sentiment_Model', 'Prediction_Type', 
             'Profit_Percentage', 'Total_Profit', 'Total_Trades', 'Final_Balance']
        ].round(4)
        
        print("\nTop Performers by Profit Percentage:")
        print("-" * 50)
        for idx, row in top_performers.iterrows():
            print(f"{row.name + 1:2d}. {row['Architecture']:10} + {row['Sentiment_Model']:8} + {row['Ticker']:4} "
                  f"| Profit: {row['Profit_Percentage']:7.2f}% | Trades: {row['Total_Trades']:3.0f} "
                  f"| Final: ${row['Final_Balance']:8.2f}")
        
        return top_performers
    
    def generate_trading_activity_analysis(self, df):
        """Analyze trading activity patterns."""
        print("\n" + "="*80)
        print("TRADING ACTIVITY ANALYSIS")
        print("="*80)
        
        # Trading activity statistics
        print("\nTrading Activity Statistics:")
        print("-" * 30)
        print(f"Average Trades per Model: {df['Total_Trades'].mean():.2f}")
        print(f"Max Trades: {df['Total_Trades'].max():.0f}")
        print(f"Min Trades: {df['Total_Trades'].min():.0f}")
        print(f"Std Dev Trades: {df['Total_Trades'].std():.2f}")
        
        # Buy vs Sell analysis
        print(f"\nBuy vs Sell Analysis:")
        print("-" * 20)
        print(f"Average Buys: {df['Buy_Count'].mean():.2f}")
        print(f"Average Sells: {df['Sell_Count'].mean():.2f}")
        print(f"Buy/Sell Ratio: {df['Buy_Count'].sum() / df['Sell_Count'].sum():.2f}")
        
        # Most active combinations
        most_active = df.nlargest(5, 'Total_Trades')[
            ['Architecture', 'Sentiment_Model', 'Ticker', 'Total_Trades', 'Profit_Percentage']
        ]
        
        print(f"\nMost Active Trading Combinations:")
        print("-" * 35)
        print(most_active.to_string(index=False))
    
    def generate_profitability_analysis(self, df):
        """Analyze profitability patterns."""
        print("\n" + "="*80)
        print("PROFITABILITY ANALYSIS")
        print("="*80)
        
        # Overall profitability
        profitable_count = df['Profitable'].sum()
        total_count = len(df)
        profit_rate = profitable_count / total_count * 100
        
        print(f"Overall Profitability:")
        print("-" * 20)
        print(f"Profitable Models: {profitable_count}/{total_count} ({profit_rate:.1f}%)")
        print(f"Average Profit (All): {df['Profit_Percentage'].mean():.4f}%")
        print(f"Average Profit (Profitable Only): {df[df['Profitable'] == 1]['Profit_Percentage'].mean():.4f}%")
        print(f"Average Loss (Losing Only): {df[df['Profitable'] == 0]['Profit_Percentage'].mean():.4f}%")
        
        # Profit distribution
        print(f"\nProfit Distribution:")
        print("-" * 18)
        print(f"Best Performance: {df['Profit_Percentage'].max():.4f}%")
        print(f"Worst Performance: {df['Profit_Percentage'].min():.4f}%")
        print(f"Median Performance: {df['Profit_Percentage'].median():.4f}%")
        print(f"75th Percentile: {df['Profit_Percentage'].quantile(0.75):.4f}%")
        print(f"25th Percentile: {df['Profit_Percentage'].quantile(0.25):.4f}%")
    
    def create_visualizations(self, df, output_dir):
        """Create visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Profit Percentage by Architecture
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='Architecture', y='Profit_Percentage')
        plt.title('Profit Percentage Distribution by Architecture', fontsize=14, fontweight='bold')
        plt.xlabel('Architecture', fontweight='bold')
        plt.ylabel('Profit Percentage (%)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'profit_by_architecture.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Profit Percentage by Sentiment Model
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='Sentiment_Model', y='Profit_Percentage')
        plt.title('Profit Percentage Distribution by Sentiment Model', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment Model', fontweight='bold')
        plt.ylabel('Profit Percentage (%)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'profit_by_sentiment.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Trading Activity vs Profitability
        plt.figure(figsize=(12, 8))
        plt.scatter(df['Total_Trades'], df['Profit_Percentage'], alpha=0.6, s=50)
        plt.xlabel('Total Trades', fontweight='bold')
        plt.ylabel('Profit Percentage (%)', fontweight='bold')
        plt.title('Trading Activity vs Profitability', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Total_Trades'], df['Profit_Percentage'], 1)
        p = np.poly1d(z)
        plt.plot(df['Total_Trades'], p(df['Total_Trades']), "r--", alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'activity_vs_profit.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance Heatmap by Architecture and Sentiment
        pivot_table = df.groupby(['Architecture', 'Sentiment_Model'])['Profit_Percentage'].mean().unstack()
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title('Average Profit Percentage Heatmap\n(Architecture vs Sentiment Model)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment Model', fontweight='bold')
        plt.ylabel('Architecture', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {output_dir}")
    
    def save_detailed_results(self, df, output_dir):
        """Save detailed results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        df.to_csv(os.path.join(output_dir, 'trading_simulation_results.csv'), index=False)
        
        # Save summary by architecture
        arch_summary = df.groupby('Architecture').agg({
            'Profit_Percentage': ['mean', 'std', 'min', 'max', 'count'],
            'Total_Profit': ['mean', 'sum'],
            'Buy_Count': 'mean',
            'Sell_Count': 'mean',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        arch_summary.columns = ['_'.join(col).strip() for col in arch_summary.columns]
        arch_summary['Success_Rate'] = (arch_summary['Profitable_sum'] / 
                                      arch_summary['Profit_Percentage_count'] * 100).round(2)
        arch_summary.to_csv(os.path.join(output_dir, 'architecture_summary.csv'))
        
        # Save summary by sentiment model
        sent_summary = df.groupby('Sentiment_Model').agg({
            'Profit_Percentage': ['mean', 'std', 'min', 'max', 'count'],
            'Total_Profit': ['mean', 'sum'],
            'Buy_Count': 'mean',
            'Sell_Count': 'mean',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        sent_summary.columns = ['_'.join(col).strip() for col in sent_summary.columns]
        sent_summary['Success_Rate'] = (sent_summary['Profitable_sum'] / 
                                      sent_summary['Profit_Percentage_count'] * 100).round(2)
        sent_summary.to_csv(os.path.join(output_dir, 'sentiment_summary.csv'))
        
        # Save summary by ticker
        ticker_summary = df.groupby('Ticker').agg({
            'Profit_Percentage': ['mean', 'std', 'min', 'max', 'count'],
            'Total_Profit': ['mean', 'sum'],
            'Buy_Count': 'mean',
            'Sell_Count': 'mean',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        ticker_summary.columns = ['_'.join(col).strip() for col in ticker_summary.columns]
        ticker_summary['Success_Rate'] = (ticker_summary['Profitable_sum'] / 
                                        ticker_summary['Profit_Percentage_count'] * 100).round(2)
        ticker_summary.to_csv(os.path.join(output_dir, 'ticker_summary.csv'))
        
        # Save model combination summary
        combination_summary = df.groupby(['Architecture', 'Sentiment_Model']).agg({
            'Profit_Percentage': ['mean', 'std', 'count'],
            'Total_Profit': 'sum',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        combination_summary.columns = ['_'.join(col).strip() for col in combination_summary.columns]
        combination_summary['Success_Rate'] = (combination_summary['Profitable_sum'] / 
                                             combination_summary['Profit_Percentage_count'] * 100).round(2)
        combination_summary = combination_summary.sort_values('Profit_Percentage_mean', ascending=False)
        combination_summary.to_csv(os.path.join(output_dir, 'model_combination_summary.csv'))
        
        # Save top performers
        top_performers = df.nlargest(20, 'Profit_Percentage')[
            ['Ticker', 'Architecture', 'Sentiment_Model', 'Prediction_Type', 
             'Profit_Percentage', 'Total_Profit', 'Total_Trades', 'Final_Balance',
             'Buy_Count', 'Sell_Count', 'Initial_Investment']
        ].round(4)
        top_performers.to_csv(os.path.join(output_dir, 'top_20_performers.csv'), index=False)
        
        # Save bottom performers
        bottom_performers = df.nsmallest(20, 'Profit_Percentage')[
            ['Ticker', 'Architecture', 'Sentiment_Model', 'Prediction_Type', 
             'Profit_Percentage', 'Total_Profit', 'Total_Trades', 'Final_Balance',
             'Buy_Count', 'Sell_Count', 'Initial_Investment']
        ].round(4)
        bottom_performers.to_csv(os.path.join(output_dir, 'bottom_20_performers.csv'), index=False)
        
        # Save trading activity analysis
        activity_stats = pd.DataFrame({
            'Metric': ['Average_Trades', 'Max_Trades', 'Min_Trades', 'Std_Trades',
                      'Average_Buys', 'Average_Sells', 'Buy_Sell_Ratio',
                      'Total_Profitable', 'Total_Models', 'Success_Rate_Percent',
                      'Avg_Profit_All', 'Avg_Profit_Profitable', 'Avg_Loss_Losing'],
            'Value': [
                df['Total_Trades'].mean(),
                df['Total_Trades'].max(),
                df['Total_Trades'].min(),
                df['Total_Trades'].std(),
                df['Buy_Count'].mean(),
                df['Sell_Count'].mean(),
                df['Buy_Count'].sum() / df['Sell_Count'].sum() if df['Sell_Count'].sum() > 0 else 0,
                df['Profitable'].sum(),
                len(df),
                (df['Profitable'].sum() / len(df) * 100),
                df['Profit_Percentage'].mean(),
                df[df['Profitable'] == 1]['Profit_Percentage'].mean(),
                df[df['Profitable'] == 0]['Profit_Percentage'].mean()
            ]
        }).round(4)
        activity_stats.to_csv(os.path.join(output_dir, 'trading_activity_stats.csv'), index=False)
        
        # Save prediction type analysis
        prediction_summary = df.groupby('Prediction_Type').agg({
            'Profit_Percentage': ['mean', 'std', 'min', 'max', 'count'],
            'Total_Profit': ['mean', 'sum'],
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        prediction_summary.columns = ['_'.join(col).strip() for col in prediction_summary.columns]
        prediction_summary['Success_Rate'] = (prediction_summary['Profitable_sum'] / 
                                            prediction_summary['Profit_Percentage_count'] * 100).round(2)
        prediction_summary.to_csv(os.path.join(output_dir, 'prediction_type_summary.csv'))
        
        # Save detailed combination analysis (Architecture + Sentiment + Ticker)
        detailed_combination = df.groupby(['Architecture', 'Sentiment_Model', 'Ticker']).agg({
            'Profit_Percentage': ['mean', 'std'],
            'Total_Profit': 'sum',
            'Total_Trades': 'mean',
            'Profitable': 'sum'
        }).round(4)
        detailed_combination.columns = ['_'.join(col).strip() for col in detailed_combination.columns]
        detailed_combination = detailed_combination.sort_values('Profit_Percentage_mean', ascending=False)
        detailed_combination.to_csv(os.path.join(output_dir, 'detailed_combination_analysis.csv'))
        
        print(f"\nAll detailed results saved to: {output_dir}")
        print("Generated CSV files:")
        print("  - trading_simulation_results.csv (main results)")
        print("  - architecture_summary.csv")
        print("  - sentiment_summary.csv") 
        print("  - ticker_summary.csv")
        print("  - model_combination_summary.csv")
        print("  - top_20_performers.csv")
        print("  - bottom_20_performers.csv")
        print("  - trading_activity_stats.csv")
        print("  - prediction_type_summary.csv")
        print("  - detailed_combination_analysis.csv")
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline."""
        print("="*80)
        print("TRADING SIMULATION COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # Load data
        if self.load_simulation_data() == 0:
            print("No simulation data found!")
            return
        
        # Create DataFrame
        df = self.create_performance_dataframe()
        
        # Generate reports
        self.generate_architecture_report(df)
        self.generate_sentiment_model_report(df)
        self.generate_ticker_report(df)
        self.generate_model_combination_report(df)
        self.generate_top_performers_report(df)
        self.generate_trading_activity_analysis(df)
        self.generate_profitability_analysis(df)
        
        # Create output directories
        output_dir = os.path.join(os.path.dirname(self.simulation_dir), 'output', 'trading_analysis')
        viz_dir = os.path.join(output_dir, 'visualizations')
        
        # Create visualizations
        self.create_visualizations(df, viz_dir)
        
        # Save detailed results
        self.save_detailed_results(df, output_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        
        return df

def main():
    """Main execution function."""
    # Define paths
    script_dir = Path(__file__).parent
    reports_dir = script_dir.parent / "reports"
    simulation_dir = reports_dir / "simulation"
    
    if not simulation_dir.exists():
        print(f"Simulation directory not found: {simulation_dir}")
        return
    
    # Create analyzer and run analysis
    analyzer = TradingAnalyzer(str(simulation_dir))
    results_df = analyzer.run_comprehensive_analysis()
    
    return results_df

if __name__ == "__main__":
    main()
