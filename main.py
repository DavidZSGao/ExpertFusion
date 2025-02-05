from data_loader import build_dataset, load_cached_data, FOMCDateBasedFetcher
from experts import MacroExpert, FundamentalExpert, NewsExpert, TechnicalExpert, RiskExpert
from moe_model import MoEModel, train_moe_model
from utils import compare_portfolio_performance, save_prediction_results
import json
from typing import List, Dict
import pandas as pd
from experts import call_gpt_factor_and_expl

def run_model(df: pd.DataFrame, start_date: str, end_date: str):
    print("[INFO] Running model...")
    
    # Dataset Info
    print("\nDataset Summary:")
    print(f"Number of stocks: {len(df['ticker'].unique())}")
    print("\nSample of data:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    
    # Rename columns if necessary
    if 'close' not in df.columns:
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close'})
        elif 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'close'})
    if 'volume' not in df.columns:
        if 'Volume' in df.columns:
            df = df.rename(columns={'Volume': 'volume'})
    
    # Debug: Calculating returns
    print("\n[DEBUG] Calculating returns...")
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    print("Sample returns:")
    print(df[['ticker', 'date', 'close', 'returns']].head())
    
    # Debug: Creating technical features
    print("\n[DEBUG] Creating technical features...")
    df['volume_ma5'] = df.groupby('ticker')['volume'].rolling(window=5).mean().reset_index(0, drop=True)
    df['price_ma5'] = df.groupby('ticker')['close'].rolling(window=5).mean().reset_index(0, drop=True)
    df['price_ma10'] = df.groupby('ticker')['close'].rolling(window=10).mean().reset_index(0, drop=True)
    print("Sample technical features:")
    print(df[['ticker', 'date', 'volume_ma5', 'price_ma5', 'price_ma10']].head())

    # Extracting PE and ROE
    print("\n[DEBUG] Extracting fundamental data...")
    def extract_pe_ratio(x):
        if isinstance(x, float):
            return x
        if isinstance(x, dict) and x:
            first_key = next(iter(x))
            if isinstance(x[first_key], dict):
                return x[first_key].get('pe_ratio', None)
            return x[first_key]
        return None

    def extract_roe(x):
        if isinstance(x, float):
            return x
        if isinstance(x, dict) and x:
            first_key = next(iter(x))
            if isinstance(x[first_key], dict):
                return x[first_key].get('roe', None)
            return x[first_key]
        return None

    df['pe'] = df['pe_ratio'].apply(extract_pe_ratio)
    df['roe'] = df['pe_ratio'].apply(extract_roe)
    df[['pe', 'roe']] = df.groupby('ticker')[['pe', 'roe']].ffill()

    # Debug: PE and ROE values
    print("Extracted PE and ROE (after forward-fill):")
    print(df[['ticker', 'date', 'pe', 'roe']].head(10))

    # Calculate volatility and risk metrics
    df['volatility'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(window=30, min_periods=5).std())
    market_returns = df[df['ticker'] == 'SPY']['returns']
    def calculate_beta(stock_returns):
        if len(stock_returns) < 5:
            return pd.Series([None] * len(stock_returns))
        rolling_cov = stock_returns.rolling(window=30, min_periods=5).cov(market_returns)
        rolling_market_var = market_returns.rolling(window=30, min_periods=5).var()
        return rolling_cov / rolling_market_var

    df['beta'] = df.groupby('ticker')['returns'].transform(calculate_beta)
    df['liquidity_ratio'] = df.groupby('ticker')['volume'].transform(lambda x: x / x.rolling(window=30, min_periods=5).mean())

    # Debug: Risk metrics
    print("Sample risk metrics:")
    print(df[['ticker', 'date', 'volatility', 'beta', 'liquidity_ratio']].head())

    # Process news data
    df['news_count'] = df['news'].apply(len)
    print("\n[DEBUG] Processing news data...")
    print("News counts:")
    print(df[['ticker', 'date', 'news_count']].head())

    # Load macro data and filter
    macro_df = load_cached_data("macro_uncertainty", start_date=start_date, end_date=end_date)
    if macro_df is None:
        raise ValueError("Macro uncertainty data could not be loaded.")
    macro_start = macro_df.index[0]
    print(f"[DEBUG] Filtering out rows with date equal to macro start date: {macro_start}")
    df = df[df['date'] != macro_start]

    # Initialize expert models
    macro_expert = MacroExpert(macro_df=macro_df, fomcFetcher=FOMCDateBasedFetcher())
    fundamental_expert = FundamentalExpert()
    news_expert = NewsExpert()
    technical_expert = TechnicalExpert()
    risk_expert = RiskExpert()

    # Collect records for training
    records = []
    for idx, row in df.iterrows():
        print(f"\n[DEBUG] Processing {row['ticker']} {str(row['date'])}:")
        expert_predictions = []
        macro_pred = macro_expert.produce_factor(row['date'], row['ticker'])
        expert_predictions.append(macro_pred)
        fund_data = {'pe': row['pe'], 'roe': row['roe']}
        fund_pred = fundamental_expert.produce_factor(fund_data)
        expert_predictions.append(fund_pred)
        news_pred = news_expert.produce_factor(row['ticker'], row['date'])
        expert_predictions.append(news_pred)
        try:
            tech_pred = technical_expert.produce_factor(df, idx)
        except Exception as e:
            print(f"[WARN] TechnicalExpert: {e} Returning default factor of 0.0.")
            tech_pred = 0.0
        expert_predictions.append(tech_pred)
        try:
            risk_pred = risk_expert.produce_factor(df, idx)
        except Exception as e:
            print(f"[WARN] RiskExpert: {e} Returning default factor of 0.0.")
            risk_pred = 0.0
        expert_predictions.append(risk_pred)

        row_data = {
            'ticker': row['ticker'],
            'date': str(row['date']),
            'macro_data': {
                'uncertainty': row['macro_data'],
                'fomc_statement': row['fomc']
            },
            'fundamental_data': {
                'pe': float(row['pe']) if pd.notnull(row['pe']) else None,
                'roe': float(row['roe']) if pd.notnull(row['roe']) else None
            },
            'technical_data': {
                'current_price': float(row['close']),
                'volume': int(row['volume']),
                'price_ma5': float(row['price_ma5']) if pd.notnull(row['price_ma5']) else None,
                'price_ma10': float(row['price_ma10']) if pd.notnull(row['price_ma10']) else None,
                'volume_ma5': float(row['volume_ma5']) if pd.notnull(row['volume_ma5']) else None
            },
            'news_data': row['news'],
            'risk_data': {
                'volatility': float(row['volatility']) if pd.notnull(row['volatility']) else None,
                'beta': float(row['beta']) if pd.notnull(row['beta']) else None,
                'liquidity_ratio': float(row['liquidity_ratio']) if pd.notnull(row['liquidity_ratio']) else None
            }
        }

        # Print the formatted data for debugging
        print("\n[DEBUG] Formatted data:")
        print(json.dumps(row_data, indent=2, default=str))

        system_msg = (
            "You are a trading assistant that analyzes market data and provides a factor score between -1 and 1 indicating bullishness/bearishness.\n"
            "Your output must be exactly two lines: 'Factor: <float>' and 'Explanation: <text>'.\n"
            "Consider all available data including fundamentals, technicals, news, and macro factors."
        )
        user_msg = f"""Please analyze this data and provide a factor score between -1 (extremely bearish) and 1 (extremely bullish):
{json.dumps(row_data, indent=2, default=str)}"""
        single_pred, explanation = call_gpt_factor_and_expl(system_msg, user_msg)
        record = {
            'ticker': row['ticker'],
            'date': row['date'],
            'expert_predictions': expert_predictions,
            'single_prediction': single_pred,
            'explanation': explanation,
            'target': row['returns'] if pd.notnull(row['returns']) else 0.0,
            'true_return': row['returns'] if pd.notnull(row['returns']) else 0.0
        }
        records.append(record)
        print(f"Expert predictions: {expert_predictions}")
        print(f"Single prediction: {single_pred}; Explanation: {explanation}.")

    # Train the MoE model
    moe_model = train_moe_model(records)
    for record in records:
        expert_preds = torch.tensor(record['expert_predictions']).unsqueeze(0)
        moe_pred = moe_model(expert_preds).item()
        record['moe_prediction'] = moe_pred

    # Compare the models and generate the portfolio comparison graph
    compare_moe_vs_single(records)

    # Portfolio Comparison: Best 20% vs Worst 20% Stocks
    compare_portfolio_performance(records, df)

    # Save prediction results to CSV
    save_prediction_results(records, 'prediction_results.csv')

    return records

def compare_moe_vs_single(records: List[Dict]):
    actual = np.array([r['true_return'] for r in records])
    moe_pred = np.array([r['moe_prediction'] for r in records])
    gpt_pred = np.array([r['single_prediction'] for r in records])
    moe_rmse = np.sqrt(mean_squared_error(actual, moe_pred))
    gpt_rmse = np.sqrt(mean_squared_error(actual, gpt_pred))
    moe_mae = mean_absolute_error(actual, moe_pred)
    gpt_mae = mean_absolute_error(actual, gpt_pred)
    moe_dir = np.mean((actual > 0) == (moe_pred > 0))
    gpt_dir = np.mean((actual > 0) == (gpt_pred > 0))
    print("\nModel Performance Comparison:")
    print("\nRoot Mean Squared Error (RMSE):")
    print(f"MoE Model: {moe_rmse:.4f}")
    print(f"GPT Model: {gpt_rmse:.4f}")
    print(f"Improvement: {((gpt_rmse - moe_rmse) / gpt_rmse * 100):.2f}%")
    print("\nMean Absolute Error (MAE):")
    print(f"MoE Model: {moe_mae:.4f}")
    print(f"GPT Model: {gpt_mae:.4f}")
    print(f"Improvement: {((gpt_mae - moe_mae) / gpt_mae * 100):.2f}%")
    print("\nDirectional Accuracy:")
    print(f"MoE Model: {moe_dir:.2%}")
    print(f"GPT Model: {gpt_dir:.2%}")
    print(f"Improvement: {((moe_dir - gpt_dir) / gpt_dir * 100):.2f}%")
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Returns', alpha=0.5)
    plt.plot(moe_pred, label='MoE Predictions', alpha=0.5)
    plt.plot(gpt_pred, label='GPT Predictions', alpha=0.5)
    plt.title('Model Predictions vs Actual Returns')
    plt.legend()
    plt.show()
    
def main():
    start_date = "2024-12-18"  # Explicit date range used for cache and processing
    end_date = "2025-01-16"
    print(f"[INFO] Using date range: {start_date} to {end_date}")
    df = build_dataset(start_date=start_date, end_date=end_date)
    records = run_model(df, start_date, end_date)
    print("\n[DEBUG] Final records sample:")
    for rec in records[:3]:
        print(json.dumps(rec, indent=2, default=str))

if __name__ == "__main__":
    main()
