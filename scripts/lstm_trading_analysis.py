import json
import os
from collections import defaultdict
import statistics

def main():
    PATH = "/mnt/c/Users/wsial/Desktop/CapstoneProject/Code/capstone-stock-prediction/reports/simulation"
    
    # Available seeds based on the file pattern
    seeds = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Store individual results for each combination to calculate means
    individual_results = defaultdict(lambda: defaultdict(list))
    
    # Store aggregated results
    results = defaultdict(lambda: {
        'total_trades': [],
        'buy_count': [],
        'sell_count': [],
        'total_profit': [],
        'profit_percentage': [],
        'final_balance': [],
        'initial_investment': []
    })

    for ticker in ['AMZN', 'AAPL', 'TSLA', 'NFLX', 'MSFT']:
        for model in ['finbert', 'roberta', 'deberta', 'lr', 'rf', 'svm']:
            print(f"Analyzing trading performance for {ticker} with {model} model")
            
            # Collect data for all seeds for this ticker-model combination
            ticker_model_data = {
                'total_trades': [],
                'buy_count': [],
                'sell_count': [],
                'total_profit': [],
                'profit_percentage': [],
                'final_balance': [],
                'initial_investment': []
            }
            
            for seed in seeds:
                filename = f'{ticker}_lstm_Binary_Price_{model}_trading_analysis_{seed}.json'
                filepath = f'{PATH}/{filename}'
                
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as file:
                            data = json.load(file)
                        
                        # Calculate total trades from buy_count + sell_count
                        total_trades = data.get('buy_count', 0) + data.get('sell_count', 0)
                        
                        # Store individual seed results
                        ticker_model_data['total_trades'].append(total_trades)
                        ticker_model_data['buy_count'].append(data.get('buy_count', 0))
                        ticker_model_data['sell_count'].append(data.get('sell_count', 0))
                        ticker_model_data['total_profit'].append(data.get('total_profit', 0.0))
                        ticker_model_data['profit_percentage'].append(data.get('profit_percentage', 0.0))
                        ticker_model_data['final_balance'].append(data.get('final_balance', 0.0))
                        ticker_model_data['initial_investment'].append(data.get('initial_investment', 1000.0))
                        
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"  Warning: Could not read {filename}: {e}")
                        continue
            
            # Calculate means for this ticker-model combination and add to overall results
            if ticker_model_data['total_trades']:  # If we have any data
                for metric in ticker_model_data:
                    if ticker_model_data[metric]:  # If list is not empty
                        mean_value = statistics.mean(ticker_model_data[metric])
                        results[model][metric].append(mean_value)
                        individual_results[model][f'{ticker}_{metric}'] = ticker_model_data[metric]
    
    # Print results with overall means across all stocks and seeds
    print("\n" + "="*80)
    print("LSTM TRADING ANALYSIS RESULTS BY SENTIMENT MODEL")
    print("(Mean across all stocks and seeds)")
    print("="*80)
    
    for model, metrics in results.items():
        if metrics['profit_percentage']:  # If we have data for this model
            mean_profit_pct = statistics.mean(metrics['profit_percentage'])
            mean_final_balance = statistics.mean(metrics['final_balance'])
            mean_total_trades = statistics.mean(metrics['total_trades'])
            mean_buy_count = statistics.mean(metrics['buy_count'])
            mean_sell_count = statistics.mean(metrics['sell_count'])
            mean_total_profit = statistics.mean(metrics['total_profit'])
            mean_initial_investment = statistics.mean(metrics['initial_investment'])
            
            # Calculate standard deviations
            std_profit_pct = statistics.stdev(metrics['profit_percentage']) if len(metrics['profit_percentage']) > 1 else 0
            std_final_balance = statistics.stdev(metrics['final_balance']) if len(metrics['final_balance']) > 1 else 0
            
            num_combinations = len(metrics['profit_percentage'])
            
            print(f"\n{model.upper()} Model:")
            print(f"  Number of Stock-Seed Combinations: {num_combinations}")
            print(f"  Mean Trades per Combination: {mean_total_trades:.1f}")
            print(f"  Mean Buys per Combination: {mean_buy_count:.1f}")
            print(f"  Mean Sells per Combination: {mean_sell_count:.1f}")
            print(f"  Mean Initial Investment: ${mean_initial_investment:.2f}")
            print(f"  Mean Final Balance: ${mean_final_balance:.2f} (±{std_final_balance:.2f})")
            print(f"  Mean Total Profit: ${mean_total_profit:.2f}")
            print(f"  Mean Profit %: {mean_profit_pct:.4f}% (±{std_profit_pct:.4f}%)")
            
            # Calculate success rate (percentage of profitable combinations)
            profitable_combinations = sum(1 for p in metrics['profit_percentage'] if p > 0)
            success_rate = (profitable_combinations / num_combinations) * 100 if num_combinations > 0 else 0
            print(f"  Success Rate: {success_rate:.1f}% ({profitable_combinations}/{num_combinations} profitable)")
    
    # Print detailed breakdown by ticker
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY TICKER")
    print("="*80)
    
    for model in results.keys():
        if results[model]['profit_percentage']:  # If we have data for this model
            print(f"\n{model.upper()} Model - By Ticker:")
            print("-" * 40)
            
            for ticker in ['AMZN', 'AAPL', 'TSLA', 'NFLX', 'MSFT']:
                ticker_profits = individual_results[model].get(f'{ticker}_profit_percentage', [])
                ticker_final_balance = individual_results[model].get(f'{ticker}_final_balance', [])
                ticker_trades = individual_results[model].get(f'{ticker}_total_trades', [])
                
                if ticker_profits:
                    mean_profit = statistics.mean(ticker_profits)
                    std_profit = statistics.stdev(ticker_profits) if len(ticker_profits) > 1 else 0
                    mean_balance = statistics.mean(ticker_final_balance)
                    mean_trades = statistics.mean(ticker_trades)
                    num_seeds = len(ticker_profits)
                    
                    print(f"  {ticker}: {mean_profit:7.3f}% (±{std_profit:.3f}%) | "
                          f"Balance: ${mean_balance:8.2f} | "
                          f"Trades: {mean_trades:4.1f} | "
                          f"Seeds: {num_seeds}")

main()
