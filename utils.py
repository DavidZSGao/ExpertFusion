import matplotlib.pyplot as plt
import json
import pandas as pd
from typing import List, Dict

def compare_portfolio_performance(records: List[Dict], df: pd.DataFrame):
    """
    Compare the MoE and GPT portfolios by buying the top 20% and shorting the bottom 20% 
    based on their factor predictions, and plotting the portfolio returns.
    """
    # Add portfolio returns
    df['moe_pred'] = df.apply(lambda row: next((r['moe_prediction'] for r in records if r['ticker'] == row['ticker'] and r['date'] == row['date']), 0.0), axis=1)
    df['gpt_pred'] = df.apply(lambda row: next((r['single_prediction'] for r in records if r['ticker'] == row['ticker'] and r['date'] == row['date']), 0.0), axis=1)

    # Sort by predictions to select best and worst 20% stocks
    df['moe_rank'] = df['moe_pred'].rank(ascending=False)
    df['gpt_rank'] = df['gpt_pred'].rank(ascending=False)
    top_20_moe = df.nlargest(int(len(df) * 0.2), 'moe_rank')
    bottom_20_moe = df.nsmallest(int(len(df) * 0.2), 'moe_rank')
    top_20_gpt = df.nlargest(int(len(df) * 0.2), 'gpt_rank')
    bottom_20_gpt = df.nsmallest(int(len(df) * 0.2), 'gpt_rank')

    # Simulate portfolio returns
    moe_returns = top_20_moe['returns'].mean() - bottom_20_moe['returns'].mean()
    gpt_returns = top_20_gpt['returns'].mean() - bottom_20_gpt['returns'].mean()
    sp500_returns = df[df['ticker'] == 'SPY']['returns'].mean()

    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['MoE Portfolio', 'GPT Portfolio', 'S&P 500'], [moe_returns, gpt_returns, sp500_returns])
    plt.title('Portfolio Performance Comparison (Best 20% / Worst 20%)')
    plt.ylabel('Average Return')
    plt.show()

def save_prediction_results(records: List[Dict], filename: str):
    # Function to save the prediction results to a CSV file
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"[INFO] Prediction results saved to {filename}")
