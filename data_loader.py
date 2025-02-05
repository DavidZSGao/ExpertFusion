import os
import time
import json
import sqlite3
import pandas as pd
import yfinance as yf
import wrds
import pickle
from functools import lru_cache
from typing import List, Dict, Tuple
import datetime
import requests
from bs4 import BeautifulSoup
import re
from config import DB_FILE, WRDS_USERNAME  # Import DB_FILE directly

###############################################
# 2) Database Access & WRDS Data
###############################################
def get_wrds_connection():
    try:
        wrds_username = os.getenv('WRDS_USERNAME')
        if wrds_username:
            print(f"[INFO] Using WRDS username: {wrds_username}")
            db = wrds.Connection(wrds_username=wrds_username)
        else:
            db = wrds.Connection()
        print("[INFO] Connected to WRDS.")
        return db
    except Exception as e:
        print(f"[ERROR] Failed to connect to WRDS: {e}")
        return None

def get_tickers_from_db(skip_wrds=False) -> List[str]:
    try:
        SP500_LIST_FILE = "sp500_list_wiki.json"
        if os.path.exists(SP500_LIST_FILE):
            with open(SP500_LIST_FILE, 'r') as f:
                sp500_data = json.load(f)
            tickers = sp500_data['tickers']
            fetch_time = sp500_data['fetch_time']
            print(f"[INFO] Loaded {len(tickers)} tickers from {SP500_LIST_FILE} (fetched at {fetch_time})")
            if skip_wrds:
                print("[INFO] Skipping WRDS connection, using tickers directly from file")
                return tickers
        else:
            raise FileNotFoundError(f"{SP500_LIST_FILE} not found. Please run fetch_sp500_list.py with VPN first.")
        db = get_wrds_connection()
        if not db:
            return tickers
        print("[INFO] Getting ticker mappings from WRDS...")
        today = pd.Timestamp.today()
        today_str = today.strftime('%Y-%m-%d')
        ticker_query = f"""
            SELECT DISTINCT n.permno, n.ticker, n.namedt, n.nameendt
            FROM crsp.msenames n
            WHERE n.ticker IN ({','.join(map(lambda x: f"'{x}'", tickers))})
              AND n.namedt <= '{today_str}'
              AND (n.nameendt >= '{today_str}' OR n.nameendt IS NULL)
            ORDER BY n.namedt DESC
        """
        ticker_df = db.raw_sql(ticker_query, date_cols=['namedt', 'nameendt'])
        print(f"[INFO] Retrieved {len(ticker_df)} ticker mappings from msenames")
        if ticker_df.empty:
            raise ValueError("No ticker mappings found in WRDS.")
        print("[INFO] Sample of ticker mappings:")
        print(ticker_df.head())
        ticker_df = ticker_df.sort_values('namedt', ascending=False).drop_duplicates('permno')
        mapped_tickers = ticker_df['ticker'].unique().tolist()
        print(f"[INFO] Retrieved {len(mapped_tickers)} unique tickers from WRDS")
        return mapped_tickers
    except Exception as e:
        print(f"[ERROR] Failed to get tickers from WRDS: {e}")
        raise

def rate_limited_query(query: str, date_cols=None, params=None) -> pd.DataFrame:
    global _LAST_QUERY_TIME
    current_time = time.time()
    time_since_last = current_time - _LAST_QUERY_TIME
    if time_since_last < _MIN_QUERY_INTERVAL:
        time.sleep(_MIN_QUERY_INTERVAL - time_since_last)
    db = get_wrds_connection()
    if not db:
        raise ConnectionError("No WRDS connection available.")
    try:
        result = db.raw_sql(query, params=params, date_cols=date_cols)
        _LAST_QUERY_TIME = time.time()
        return result
    except Exception as e:
        print(f"[ERROR] WRDS query failed: {e}")
        raise

@lru_cache(maxsize=100)
def get_fundamentals_wrds(ticker: str, date: datetime.date) -> Dict:
    try:
        wrds_data = load_cached_data("wrds", date=date.strftime('%Y-%m-%d'))
        if wrds_data and ticker in wrds_data:
            return wrds_data[ticker].get(date.isoformat(), {})
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to get fundamental data: {e}")
        raise

def get_macro_uncertainty(start_date: str, end_date: str, use_cache=True) -> pd.DataFrame:
    cache_file = f'macro_uncertainty_cache_{start_date}_{end_date}.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    try:
        with open(cache_file, 'rb') as f:
            macro_df = pickle.load(f)
        print(f"[INFO] Loaded macro_uncertainty data from cache: {cache_file}")
        if not isinstance(macro_df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            if isinstance(macro_df.index, pd.RangeIndex):
                print(f"[DEBUG] Reconstructing macro_uncertainty index using start_date {start_date} and end_date {end_date}")
                new_index = pd.date_range(start=start_date, end=end_date, freq='D').date
                macro_df.index = new_index
            else:
                raise TypeError(f"Unexpected macro_df index type: {type(macro_df.index)}. Full index: {macro_df.index}")
        return macro_df
    except Exception as e:
        print(f"[ERROR] Failed to load macro_uncertainty cache: {e}")
        raise

def cache_wrds_data(date: str) -> Dict:
    print("[INFO] Caching WRDS data...")
    data = {}
    try:
        db = get_wrds_connection()
        if not db:
            raise ConnectionError("No WRDS connection available for caching.")
        tickers = get_tickers_from_db()
        for ticker in tickers:
            try:
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                fundamentals = get_fundamentals_wrds(ticker, date_obj)
                if fundamentals:
                    data[ticker] = {date: fundamentals}
            except Exception as e:
                print(f"[ERROR] Failed to get WRDS data for {ticker}: {e}")
                continue
        if data:
            print(f"[INFO] Saving WRDS data for {len(data)} tickers")
            with open(f'wrds_cache_{date}.pkl', 'wb') as f:
                pickle.dump(data, f)
        else:
            print("[WARN] No WRDS data to cache")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to cache WRDS data: {e}")
        raise

def cache_yfinance_data(tickers: List[str], start_date: str, end_date: str) -> Dict:
    print("[INFO] Caching yfinance data...")
    data = {}
    for ticker in tickers:
        try:
            df = get_market_data(ticker, start_date, end_date)
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            print(f"[ERROR] yfinance caching failed for {ticker}: {e}")
            continue
    if data:
        print(f"[INFO] Saving yfinance data for {len(data)} tickers")
        with open(f'yfinance_cache_{start_date}_{end_date}.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        print("[WARN] No yfinance data to cache")
    return data

def load_cached_data(data_type: str, date: str = None, start_date: str = None, end_date: str = None):
    if data_type == "wrds":
        cache_file = f'wrds_cache_{date}.pkl'
    elif data_type == "yfinance":
        cache_file = f'yfinance_cache_{start_date}_{end_date}.pkl'
    elif data_type == "macro_uncertainty":
        cache_file = f'macro_uncertainty_cache_{start_date}_{end_date}.pkl'
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"[INFO] Loaded {data_type} data from cache: {cache_file}")
        if data_type == "macro_uncertainty":
            if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                if isinstance(data.index, pd.RangeIndex):
                    print(f"[DEBUG] Reconstructing macro_uncertainty index using start_date {start_date} and end_date {end_date}")
                    new_index = pd.date_range(start=start_date, end=end_date, freq='D').date
                    data.index = new_index
                else:
                    raise TypeError(f"Unexpected macro_df index type: {type(data.index)}. Full index: {data.index}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load {data_type} cache: {e}")
        raise

def get_news_from_db(ticker: str, date: datetime.date) -> List[str]:
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT headlines FROM news_storage WHERE ticker = ? AND date = ?", (ticker, date.isoformat()))
        result = c.fetchone()
        conn.close()
        if result:
            return result[0].split(" || ")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to get news from DB: {e}")
        raise

def get_date_range_from_db() -> Tuple[datetime.date, datetime.date]:
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT MIN(date), MAX(date) FROM news_storage")
        start_date_str, end_date_str = c.fetchone()
        conn.close()
        if start_date_str and end_date_str:
            return (datetime.date.fromisoformat(start_date_str),
                    datetime.date.fromisoformat(end_date_str))
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)
        return start_date, end_date
    except Exception as e:
        print(f"[ERROR] Failed to get date range from DB: {e}")
        raise

def get_fundamentals_av(ticker: str) -> Dict:
    try:
        db = get_wrds_connection()
        if not db:
            raise ConnectionError("No WRDS connection available.")
        print(f"[INFO] Querying fundamental data for {ticker} from WRDS...")
        query = """
            SELECT c.gvkey, c.datadate, 
                   c.prccm/c.ceqq AS pe_ratio,
                   c.ibq/c.ceqq AS roe
            FROM comp.fundq AS c
            JOIN crsp.ccmxpf_linktable AS l ON c.gvkey = l.gvkey
            JOIN crsp.dsenames AS d ON l.lpermno = d.permno
            WHERE d.ticker = %s AND c.ceqq > 0
            ORDER BY c.datadate DESC
            LIMIT 1
        """
        fund_data = rate_limited_query(query, date_cols=['datadate'], params=(ticker,))
        if fund_data.empty:
            raise ValueError(f"No fundamental data found for {ticker}")
        result = {"pe_ratio": str(fund_data.iloc[0]['pe_ratio']),
                  "roe": str(fund_data.iloc[0]['roe'])}
        return result
    except Exception as e:
        print(f"[ERROR] Failed to get fundamental data from WRDS: {e}")
        raise

###############################################
# 3) FOMC Data
###############################################
class FOMCDateBasedFetcher:
    BASE_URL = "https://www.federalreserve.gov"
    CAL_URL  = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
    def __init__(self):
        self.all_meetings = self.parse_calendars_for_all_years()
        self.all_meetings.sort(key=lambda x: x["end_date"])
        for m in self.all_meetings:
            m["statement_text"] = None
    def fetch_fomc_calendars_page(self) -> str:
        try:
            resp = requests.get(self.CAL_URL)
            if not resp.ok:
                raise ValueError(f"FOMC calendars fetch failed with HTTP {resp.status_code}")
            return resp.text
        except Exception as e:
            print("[ERROR] fetch_fomc_calendars_page =>", e)
            raise
    def fetch_fomc_statement_text(self, link_url: str) -> str:
        try:
            resp = requests.get(link_url)
            if not resp.ok:
                raise ValueError(f"Statement fetch failed with HTTP {resp.status_code}")
            soup = BeautifulSoup(resp.text, "html.parser")
            main_div = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
            if not main_div:
                raise ValueError("FOMC statement div not found")
            return main_div.get_text(separator="\n", strip=True)
        except Exception as e:
            print("[ERROR] fetch_fomc_statement_text =>", e)
            raise
    def parse_meeting_divs_for_year(self, html: str, year: int) -> list:
        soup = BeautifulSoup(html, "html.parser")
        meeting_divs = soup.find_all("div", class_=re.compile(r"fomc-meeting"))
        results = []
        for div in meeting_divs:
            raw_text = div.get_text(" ", strip=True)
            m = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:-(\d{1,2}))?",
                raw_text
            )
            if not m:
                continue
            month_str = m.group(1)
            day1 = int(m.group(2))
            day2 = m.group(3)
            month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                         'May': 5, 'June': 6, 'July': 7, 'August': 8,
                         'September': 9, 'October': 10, 'November': 11, 'December': 12}
            month_int = month_map.get(month_str, 1)
            start_dt = datetime.date(year, month_int, day1)
            end_dt   = datetime.date(year, month_int, int(day2)) if day2 else start_dt
            link_tag = div.find("a", href=re.compile(r"/newsevents/pressreleases/monetary.*\.htm"))
            link_href = ""
            if link_tag and link_tag.has_attr("href"):
                link_href = link_tag["href"]
                if link_href.startswith("/"):
                    link_href = self.BASE_URL + link_href
            snippet = raw_text[:120]
            results.append({"start_date": start_dt, "end_date": end_dt, "link": link_href, "snippet": snippet})
        return results
    def parse_calendars_for_all_years(self) -> list:
        out = []
        full_html = self.fetch_fomc_calendars_page()
        if not full_html:
            raise ValueError("No HTML received from FOMC calendars page.")
        soup = BeautifulSoup(full_html, "html.parser")
        panels = soup.find_all("div", class_="panel panel-default")
        for panel in panels:
            heading = panel.find("div", class_="panel-heading")
            if not heading:
                continue
            heading_txt = heading.get_text(" ", strip=True)
            m = re.search(r"(\d{4}) FOMC Meetings", heading_txt)
            if not m:
                continue
            year = int(m.group(1))
            panel_html = str(panel)
            subset = self.parse_meeting_divs_for_year(panel_html, year)
            out.extend(subset)
        return out
    def get_most_recent_fomc_for(self, date_val: datetime.date) -> str:
        valid = [m for m in self.all_meetings if m["end_date"] <= date_val]
        if not valid:
            raise ValueError(f"No FOMC meeting found before {date_val}")
        valid_desc = sorted(valid, key=lambda x: x["end_date"], reverse=True)
        for meeting in valid_desc:
            link = meeting["link"]
            if not link:
                continue
            if meeting["statement_text"]:
                return meeting["statement_text"]
            text = self.fetch_fomc_statement_text(link)
            if text:
                meeting["statement_text"] = text
                return text
        raise ValueError(f"No FOMC statement could be retrieved for date {date_val}")

###############################################
# 4) Market Data
###############################################
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    return df

def rename_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rename_dict = {}
    suffix = f"_{ticker}"
    for c in df.columns:
        if c.endswith(suffix):
            newc = c.replace(suffix, "")
            rename_dict[c] = newc
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    return df

def get_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        yf_ticker = ticker
        if any(x in ticker for x in ['.', '-', '/']):
            yf_ticker = ticker.replace('-', '.').replace('/', '.')
        print(f"[INFO] Getting market data for {ticker} from {start_date} to {end_date}")
        df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            print(f"[WARN] No data returned for {ticker}")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] + ' ' + col[1] if col[1] else col[0] for col in df.columns]
        col_map = {'Adj Close': 'close', 'Adj_Close': 'close', 'adj_close': 'close',
                   'Volume': 'volume', 'volume': 'volume'}
        df = df.rename(columns=col_map)
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'date'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        return df
    except Exception as e:
        print(f"[ERROR] Failed to get market data for {ticker}: {e}")
        raise
    
###############################################
# 7) Build Dataset
###############################################
def build_dataset(tickers: List[str] = None, start_date: str = None, end_date: str = None, fomcFetcher=None):
    print("[INFO] Building dataset...")
    print("[INFO] Loading tickers from file...")
    with open("sp500_list_wiki.json", "r") as f:
        data = json.load(f)
        tickers = data["tickers"]
        print(f"[INFO] Loaded {len(tickers)} tickers from sp500_list_wiki.json (fetched at {data['fetch_time']})")
    wrds_data = load_cached_data("wrds", date=start_date)
    yf_data = load_cached_data("yfinance", start_date=start_date, end_date=end_date)
    macro_data = load_cached_data("macro_uncertainty", start_date=start_date, end_date=end_date)
    if fomcFetcher is None:
        fomcFetcher = FOMCDateBasedFetcher()
    dfs = []
    for ticker in yf_data:
        df = yf_data[ticker].copy()
        if 'close' not in df.columns:
            if 'Close' in df.columns:
                df = df.rename(columns={'Close': 'close'})
            elif 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'close'})
        if 'volume' not in df.columns:
            if 'Volume' in df.columns:
                df = df.rename(columns={'Volume': 'volume'})
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df.index).date
        df['pe_ratio'] = df['date'].apply(lambda x: wrds_data.get(ticker, {}).get(x.strftime('%Y-%m-%d'), {}))
        df['news'] = df['date'].apply(lambda x: get_news_from_db(ticker, x))
        if isinstance(macro_data, pd.DataFrame) and not macro_data.empty:
            df['macro_data'] = df['date'].apply(lambda x: macro_data.loc[pd.Timestamp(x).strftime('%Y-%m-%d')]['value']
                                                 if pd.Timestamp(x).strftime('%Y-%m-%d') in macro_data.index
                                                 else macro_data['value'].iloc[-1])
        else:
            df['macro_data'] = 0
        if fomcFetcher is not None:
            df['fomc'] = df['date'].apply(lambda x: fomcFetcher.get_most_recent_fomc_for(x))
        else:
            df['fomc'] = ''
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    print("\n[DEBUG] DataFrame info:")
    print(combined_df.info())
    print("\n[DEBUG] First few rows:")
    print(combined_df.head())
    return combined_df
