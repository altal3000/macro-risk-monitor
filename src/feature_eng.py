import os
import logging
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

S3_BUCKET = "macro-risk-monitor"
s3 = boto3.client("s3")
TODAY = datetime.today().strftime("%Y-%m-%d")

# --- S3 read helper ---
def read_from_s3(source: str) -> pd.DataFrame:
    key = f"raw/{source}/{TODAY}.parquet"
    local_path = f"/tmp/{source}_{TODAY}.parquet"
    try:
        s3.download_file(S3_BUCKET, key, local_path)
        df = pd.read_parquet(local_path)
        logger.info(f"Loaded {source}: {df.shape}")
        return df
    except Exception as e:
        logger.warning(f"Could not load {source}: {e}")
        return pd.DataFrame()

# --- Compute the price features ---
def compute_price_features(df_yf: pd.DataFrame, df_fred: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df_yf.index)
    prices = df_yf.clip(lower=1e-6)

    # Brent from FRED — reindex to yfinance index, ffill gaps
    brent = df_fred["brent"].reindex(df_yf.index).ffill().clip(lower=1e-6)

    f["brent_ret_1d"]     = np.log(brent / brent.shift(1))
    f["brent_ret_5d"]     = np.log(brent / brent.shift(5))
    f["wti_ret_1d"]       = np.log(prices["CL=F"] / prices["CL=F"].shift(1))
    f["brent_wti_spread"] = brent - prices["CL=F"]
    f["brent_vol_20d"]    = f["brent_ret_1d"].rolling(20).std()
    f["ng_ret_1d"]        = np.log(prices["NG=F"] / prices["NG=F"].shift(1))
    f["xle_ret_1d"]       = np.log(prices["XLE"] / prices["XLE"].shift(1))
    f["xle_brent_ratio"]  = prices["XLE"] / brent
    f["ovx_zscore"]       = (prices["^OVX"] - prices["^OVX"].rolling(60).mean()) / prices["^OVX"].rolling(60).std()
    f["vix_zscore"]       = (prices["^VIX"] - prices["^VIX"].rolling(60).mean()) / prices["^VIX"].rolling(60).std()

    logger.info(f"Price features: {f.shape}")
    return f

# --- Compute the macro features ---
def compute_macro_features(df_fred: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df_fred.index)

    f["yield_curve"]         = df_fred["yield_curve"]
    f["yield_curve_chg_20d"] = df_fred["yield_curve"].diff(20)
    f["hy_spread"]           = df_fred["hy_spread"]
    f["hy_spread_chg_5d"]    = df_fred["hy_spread"].diff(5)
    f["fed_rate"]            = df_fred["fed_rate"]

    cpi_monthly = df_fred["cpi"].dropna().pct_change(12) * 100
    f["cpi_yoy"] = cpi_monthly.reindex(df_fred.index).ffill()

    dxy = df_fred["dxy"].dropna()
    dxy_z = (dxy - dxy.rolling(60).mean()) / dxy.rolling(60).std()
    f["dxy_zscore"] = dxy_z.reindex(df_fred.index).ffill()

    logger.info(f"Macro features: {f.shape}")
    return f

# --- Compute geopolitical features ---
def compute_geo_features(df_gpr: pd.DataFrame, ovx_zscore: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=df_gpr.index)

    days_stale = int(df_gpr["gpr_days_stale"].iloc[-1])
    is_stale = bool(df_gpr["gpr_is_stale"].iloc[-1])

    gpr_weight = 0.95 ** days_stale
    gpr_decayed = gpr_weight * df_gpr["GPR_AI"] + (1 - gpr_weight) * 100
    ovx_component = (1 - gpr_weight) * ovx_zscore.reindex(df_gpr.index) * 30

    f["geo_signal"]     = gpr_decayed + ovx_component
    f["gpr_oil_zscore"] = (
        (df_gpr["GPR_OIL"] - df_gpr["GPR_OIL"].rolling(60).mean())
        / df_gpr["GPR_OIL"].rolling(60).std()
    )
    f["gpr_is_stale"]   = is_stale
    f["gpr_days_stale"] = days_stale

    logger.info(f"Geo features: {f.shape}")
    return f

# --- Align all features ---
def align_features(
    price_f: pd.DataFrame,
    macro_f: pd.DataFrame,
    geo_f: pd.DataFrame
) -> pd.DataFrame:
    combined = pd.concat([price_f, macro_f, geo_f], axis=1, sort=True)
    combined = combined.sort_index()
    combined = combined.ffill()
    combined = combined[combined.index >= "2007-01-01"]
    combined["is_trading_day"]     = True
    combined["data_quality_score"] = 1.0
    combined["data_quality_flags"] = ""
    logger.info(f"Aligned features: {combined.shape}")
    return combined

# --- Load existing features ---
def load_existing_features() -> pd.DataFrame:
    try:
        s3.download_file(S3_BUCKET, "features/features_window.parquet", "/tmp/features_window.parquet")
        df = pd.read_parquet("/tmp/features_window.parquet")
        logger.info(f"Loaded existing features: {df.shape}")
        return df
    except Exception as e:
        logger.info("No existing features found — starting fresh")
        return pd.DataFrame()

# --- Engineer features ---
if __name__ == "__main__":
    logger.info(f"Starting feature engineering — {TODAY}")

    df_yf   = read_from_s3("yfinance")
    df_fred = read_from_s3("fred")
    df_gpr  = read_from_s3("gpr")

    price_f = compute_price_features(df_yf, df_fred)
    macro_f = compute_macro_features(df_fred)
    geo_f   = compute_geo_features(df_gpr, price_f["ovx_zscore"])

    features = align_features(price_f, macro_f, geo_f)

    # Merge with existing — only append new dates, never overwrite
    existing = load_existing_features()
    if not existing.empty:
        new_dates = features.index.difference(existing.index)
        if len(new_dates) > 0:
            features = pd.concat([existing, features.loc[new_dates]]).sort_index()
            logger.info(f"Merged features: {features.shape}")
        else:
            features = existing.copy()
            logger.info("No new dates — using existing features")

    features.to_parquet("/tmp/features_window.parquet")
    s3.upload_file("/tmp/features_window.parquet", S3_BUCKET, "features/features_window.parquet")
    logger.info(f"Saved to s3://{S3_BUCKET}/features/features_window.parquet")

    logger.info("Feature engineering complete")