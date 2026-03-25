import os
import logging
from datetime import datetime

import boto3
import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred
import exchange_calendars as xcals
from dotenv import load_dotenv
import argparse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# AWS
S3_BUCKET = "macro-risk-monitor"
s3 = boto3.client("s3")

# Date
TODAY = datetime.today().strftime("%Y-%m-%d")

# Market calendat check
def is_trading_day(date: str) -> bool:
    cal = xcals.get_calendar("XNYS")
    return cal.is_session(pd.Timestamp(date))

TRADING_DAY = is_trading_day(TODAY)

if not TRADING_DAY:
    logger.info(f"{TODAY} is not a trading day — prices will be forward-filled")

# Add and fetch tickers
TICKERS = ["BZ=F", "CL=F", "NG=F", "XLE", "XOP", "^OVX", "^VIX"]

def fetch_yfinance(period: str = "5d") -> pd.DataFrame:
    logger.info(f"Fetching yfinance data (period={period})...")
    try:
        df = yf.download(TICKERS, period=period, interval="1d", auto_adjust=True, progress=False)
        df = df["Close"]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        logger.info(f"yfinance: {len(df)} rows fetched")
        return df
    except Exception as e:
        logger.warning(f"yfinance fetch failed: {e}")
        return pd.DataFrame()
    
# Upload to S3
def save_to_s3(df: pd.DataFrame, source: str) -> None:
    if df.empty:
        logger.warning(f"Skipping S3 upload for {source} — empty dataframe")
        return
    
    key = f"raw/{source}/{TODAY}.parquet"
    df.to_parquet("/tmp/temp.parquet")
    s3.upload_file("/tmp/temp.parquet", S3_BUCKET, key)
    logger.info(f"Saved to s3://{S3_BUCKET}/{key}")

# Fetch FRED data
FRED_API_KEY = os.getenv("FRED_API_KEY")

FRED_SERIES = {
    "fed_rate":    "DFF",
    "yield_curve": "T10Y2Y",
    "dxy":         "DTWEXBGS",
    "hy_spread":   "BAMLH0A0HYM2",
    "cpi":         "CPIAUCSL",
    "brent":       "DCOILBRENTEU"
}

def fetch_fred() -> pd.DataFrame:
    logger.info("Fetching FRED data...")
    try:
        fred = Fred(api_key=FRED_API_KEY)
        series = {}
        for name, series_id in FRED_SERIES.items():
            s = fred.get_series(series_id, observation_start="2007-01-01")
            series[name] = s
        df = pd.DataFrame(series)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        logger.info(f"FRED: {len(df)} rows fetched")
        return df
    except Exception as e:
        logger.warning(f"FRED fetch failed: {e}")
        return pd.DataFrame()

# Fetch GPR data
GPR_URL = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"

def fetch_gpr() -> pd.DataFrame:
    logger.info("Fetching AI-GPR data...")
    try:
        df = pd.read_csv(GPR_URL)
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        
        days_stale = (pd.Timestamp.today() - df.index.max()).days
        df["gpr_days_stale"] = days_stale
        df["gpr_is_stale"] = days_stale > 10
        
        logger.info(f"GPR: {len(df)} rows fetched, {days_stale} days stale")
        return df
    except Exception as e:
        logger.warning(f"GPR fetch failed: {e}")
        return pd.DataFrame()

# --- Ingest data ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Pull full history")
    args = parser.parse_args()

    period = "max" if args.backfill else "5d"
    logger.info(f"Starting ingest — {TODAY} (backfill={args.backfill})")

    df_yf = fetch_yfinance(period=period)
    save_to_s3(df_yf, "yfinance")

    df_fred = fetch_fred()
    save_to_s3(df_fred, "fred")

    df_gpr = fetch_gpr()
    save_to_s3(df_gpr, "gpr")

    logger.info("Ingest complete")