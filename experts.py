import datetime
from typing import Dict, Tuple
from data_loader import get_news_from_db, FOMCDateBasedFetcher
import openai
import json
import re
import pandas as pd
from config import OPENAI_API_KEY
# or OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client with API key from config
client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

def call_gpt_factor_and_expl(system_msg: str, user_msg: str) -> Tuple[float, str]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=150
        )
        text = response.choices[0].message.content
        # Debug: print raw GPT response
        print(f"[DEBUG] Raw GPT response:\n{text}\n")
        factor_match = re.search(r"Factor:\s*(-?\d*\.?\d+)", text)
        expl_match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)
        factor = float(factor_match.group(1)) if factor_match else 0.0
        explanation = expl_match.group(1).strip() if expl_match else ""
        if not explanation:
            print("[WARN] No explanation parsed from GPT response.")
        return factor, explanation
    except Exception as e:
        print(f"[ERROR] GPT call failed: {e}")
        return 0.0, ""

    
# Define Expert classes
class MacroExpert:
    """Expert for macro-economic factors."""
    def __init__(self, macro_df: pd.DataFrame, fomcFetcher: FOMCDateBasedFetcher):
        self.macro_df = macro_df
        self.fomcFetcher = fomcFetcher
    def produce_factor(self, dt_val: datetime.date, ticker: str) -> float:
        try:
            if self.macro_df is None or self.macro_df.empty:
                raise ValueError("Macro data is missing or empty.")
            first_index = self.macro_df.index[0]
            print(f"[DEBUG] Macro data index type: {type(first_index)}, value: {first_index}")
            if isinstance(first_index, pd.Timestamp):
                hist_data = self.macro_df[self.macro_df.index.map(lambda x: x.date()) <= dt_val]
            elif isinstance(first_index, datetime.date):
                hist_data = self.macro_df[self.macro_df.index <= dt_val]
            else:
                raise TypeError(f"Unexpected macro_df index type: {type(first_index)}. Full index: {self.macro_df.index}")
            if len(hist_data) < 2:
                raise ValueError(f"Not enough macro data history before {dt_val}.")
            current = float(hist_data.iloc[-1]['value'])
            previous = float(hist_data.iloc[-2]['value'])
            fomc_text = self.fomcFetcher.get_most_recent_fomc_for(dt_val)
            fomc_snippet = fomc_text[:200] if fomc_text else "No FOMC statement"
            sys = (
                "You are a macro analysis expert. Consider how GDP and monetary policy specifically impact this company:\n"
                "1. Analyze the company's industry, business model, and macro sensitivities, considering both short and long-term impacts.\n"
                "2. Evaluate how current GDP and FOMC stance affect this specific business.\n"
                "3. Consider factors like discretionary vs essential products, consumer financing, interest rates, "
                "supply chain, global trade, currency exposure, and international revenue.\n"
                "4. Based on these factors, determine whether the overall macro environment is bullish or bearish for this company.\n"
                "5. Output a factor between -1 (very negative) and +1 (very positive), reflecting the bullish or bearish nature.\n"
                "Format => Factor: <float>\n"
                "Explanation: <text>"
            )
            usr = f"Company: {ticker}\nGDP={current:.2f} vs {previous:.2f}\nFOMC: {fomc_snippet}"
            fac, _ = call_gpt_factor_and_expl(sys, usr)
            return fac
        except Exception as e:
            print(f"[ERROR] MacroExpert failed: {e}")
            raise

class FundamentalExpert:
    """Expert for fundamental analysis."""
    def produce_factor(self, fundamentals: Dict) -> float:
        try:
            if not fundamentals:
                raise ValueError("No fundamental data provided.")
            pe = fundamentals.get('pe', '')
            roe = fundamentals.get('roe', '')
            sys = (
                "You are an expert in fundamental analysis. Evaluate the company's financial health based on the following:\n"
                "1. P/E ratio: Is the company undervalued (bullish) or overvalued (bearish)? Consider industry averages and market conditions.\n"
                "2. ROE: Is the company generating strong returns on equity? Does it indicate a healthy business model?\n"
                "3. Based on these financial metrics, determine if the stock is bullish or bearish.\n"
                "Output a factor between -1 (very negative) and +1 (very positive), with an explanation based on the metrics.\n"
                "Format => Factor: <float>\n"
                "Explanation: <text>"
            )
            usr = f"P/E={pe}, ROE={roe}"
            fac, _ = call_gpt_factor_and_expl(sys, usr)
            return fac
        except Exception as e:
            print(f"[ERROR] FundamentalExpert failed: {e}")
            raise

class NewsExpert:
    def produce_factor(self, ticker: str, date: datetime.date) -> float:
        try:
            headlines = get_news_from_db(ticker, date)
            if not headlines:
                print(f"[WARN] No news found for {ticker} on {date}. Returning default factor of 0.0.")
                return 0.0
            short = "\n".join(headlines[:3])
            sys = (
                "You are an expert in news sentiment analysis. Consider the news headlines for the company and evaluate:\n"
                "1. Are the headlines generally positive, neutral, or negative in terms of market sentiment?\n"
                "2. How does the news affect investor sentiment towards the company? Is the news likely to drive the stock price up or down?\n"
                "3. Based on your analysis, determine whether the overall sentiment is bullish or bearish for the stock.\n"
                "4. Output a factor between -1 (very negative) and +1 (very positive) reflecting the sentiment.\n"
                "Format => Factor: <float>\n"
                "Explanation: <text>"
            )
            usr = f"{short}"
            fac, _ = call_gpt_factor_and_expl(sys, usr)
            return fac
        except Exception as e:
            print(f"[ERROR] NewsExpert failed: {e}")
            raise

class TechnicalExpert:
    def produce_factor(self, df: pd.DataFrame, global_idx: int) -> float:
        try:
            # Filter for the current ticker and reset index.
            row = df.iloc[global_idx]
            ticker = row['ticker']
            group = df[df['ticker'] == ticker].reset_index(drop=True)
            positions = group.index[group['date'] == row['date']]
            if len(positions) == 0:
                raise ValueError(f"Could not find row for ticker {ticker} on {row['date']} in its group.")
            pos = positions[0]
            if pos < 4:
                print(f"[WARN] Not enough bars for technical analysis for ticker {ticker} on {row['date']}. Returning default factor of 0.0.")
                return 0.0
            pct_changes = group['close'].pct_change().values
            w5 = pct_changes[pos-4:pos+1]
            s = ", ".join(f"{x:.4f}" for x in w5 if not np.isnan(x))
            sys = (
                "You are an expert in technical analysis. Evaluate the following technical indicators for the stock:\n"
                "1. How do recent price and volume patterns reflect the overall trend in the market?\n"
                "2. Is the stock showing bullish or bearish signals based on technical indicators such as moving averages, momentum, or volume trends?\n"
                "3. Based on this analysis, is the stock in a bullish or bearish trend, and what is your overall sentiment?\n"
                "Output a factor between -1 (very negative) and +1 (very positive), representing the technical sentiment.\n"
                "Format => Factor: <float>\n"
                "Explanation: <text>"
            )
            usr = f"Last5 returns => {s}"
            fac, _ = call_gpt_factor_and_expl(sys, usr)
            return fac
        except Exception as e:
            print(f"[ERROR] TechnicalExpert failed: {e}")
            raise

class RiskExpert:
    def produce_factor(self, df: pd.DataFrame, global_idx: int) -> float:
        try:
            if global_idx < 5 or 'close' not in df.columns:
                raise ValueError("Not enough data for risk analysis.")
            rets = df['close'].pct_change().dropna().values
            w_5 = rets[global_idx-4:global_idx+1]
            stv = np.nanstd(w_5)
            sys = (
                "You are an expert in risk analysis. Evaluate the following risk metrics for the stock:\n"
                "1. What does the recent volatility and beta tell you about the stock's risk profile?\n"
                "2. Is the stock showing signs of higher volatility (bearish) or stability (bullish)? Consider if the risk is manageable or excessive.\n"
                "3. Based on these risk factors, do you consider the stock to be a higher or lower risk, and how does this affect its bullish or bearish nature?\n"
                "4. Output a factor between -1 (very negative) and +1 (very positive), reflecting the risk sentiment.\n"
                "Format => Factor: <float>\n"
                "Explanation: <text>"
            )
            usr = f"5-day std dev = {stv:.4f}"
            fac, _ = call_gpt_factor_and_expl(sys, usr)
            return fac
        except Exception as e:
            print(f"[ERROR] RiskExpert failed: {e}")
            raise