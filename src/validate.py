import os
import sys
import logging
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import exchange_calendars as xcals

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

S3_BUCKET = "macro-risk-monitor"
s3 = boto3.client("s3")
TODAY = datetime.today().strftime("%Y-%m-%d")

CRITICAL = "CRITICAL"
WARNING  = "WARNING"
INFO     = "INFO"

# S3 read
def load_features() -> pd.DataFrame:
    s3.download_file(S3_BUCKET, "features/features_window.parquet", "/tmp/features_window.parquet")
    df = pd.read_parquet("/tmp/features_window.parquet")
    logger.info(f"Loaded features: {df.shape}")
    return df

# Calendar check
def check_trading_day() -> bool:
    cal = xcals.get_calendar("XNYS")
    is_trading = cal.is_session(pd.Timestamp(TODAY))
    if not is_trading:
        logger.info(f"{TODAY} is not a trading day — price checks relaxed")
    return is_trading

# Critical checks
def check_critical(df: pd.DataFrame, is_trading_day: bool) -> list:
    failures = []

    # Schema check
    expected_cols = [
        "brent_ret_1d", "brent_ret_5d", "wti_ret_1d", "brent_wti_spread",
        "brent_vol_20d", "ng_ret_1d", "xle_ret_1d", "xle_brent_ratio",
        "ovx_zscore", "vix_zscore", "yield_curve", "yield_curve_chg_20d",
        "dxy_zscore", "hy_spread", "hy_spread_chg_5d", "fed_rate",
        "cpi_yoy", "geo_signal", "gpr_oil_zscore",
        "gpr_is_stale", "gpr_days_stale"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        failures.append(f"SCHEMA: missing columns {missing}")

    # Zero rows
    if len(df) == 0:
        failures.append("COMPLETENESS: zero rows")

    # Multi-day gap on trading days
    if is_trading_day:
        last_date = df.index.max()
        gap = (pd.Timestamp(TODAY) - last_date).days
        if gap > 2:
            failures.append(f"GAP: last date is {last_date.date()}, gap={gap} days")

    # Cross-source consistency
    if is_trading_day and len(df) > 0:
        latest = df.iloc[-1]
        brent_move = abs(latest.get("brent_ret_1d", 0))
        ovx_z = abs(latest.get("ovx_zscore", 0))
        vix_z = abs(latest.get("vix_zscore", 0))
        if brent_move > 0.10 and ovx_z < 1.0 and vix_z < 1.0:
            failures.append(f"CROSS_SOURCE: Brent moved {brent_move:.1%} but vol indices flat")

    return failures

# Warning checks
def check_warnings(df: pd.DataFrame) -> list:
    warnings = []

    # GPR staleness
    gpr_days_stale = int(df["gpr_days_stale"].iloc[-1])
    if gpr_days_stale > 10:
        warnings.append(f"GPR_STALE: {gpr_days_stale} days stale — OVX fallback active")

    # Null threshold — any feature column > 5% null in last 60 rows
    recent = df.iloc[-60:]
    for col in df.columns:
        null_pct = recent[col].isnull().mean()
        if null_pct > 0.05:
            warnings.append(f"NULLS: {col} is {null_pct:.0%} null in last 60 rows")

    # Single source missing — all price features null today
    latest = df.iloc[-1]
    price_cols = ["brent_ret_1d", "wti_ret_1d", "ng_ret_1d"]
    if all(pd.isna(latest[c]) for c in price_cols):
        warnings.append("SOURCE_MISSING: all price features null today")

    return warnings

# Quality score
def compute_quality_score(warnings: list) -> float:
    score = 1.0 - (len(warnings) * 0.1)
    return max(0.0, score)

# Main gate
def run_gate(df: pd.DataFrame, failures: list, warnings: list) -> pd.DataFrame:
    # Write quality metadata back to dataframe
    score = compute_quality_score(warnings)
    flags = "|".join(warnings) if warnings else ""

    df["data_quality_score"] = score
    df["data_quality_flags"] = flags
    df["is_trading_day"] = check_trading_day()

    return df

# Save to S3
def save_features(df: pd.DataFrame) -> None:
    local_path = "/tmp/features_window_validated.parquet"
    df.to_parquet(local_path)
    s3.upload_file(local_path, S3_BUCKET, "features/features_window.parquet")
    logger.info(f"Saved validated features to s3://{S3_BUCKET}/features/features_window.parquet")

# --- Run validation gate ---
if __name__ == "__main__":
    logger.info(f"Starting validation — {TODAY}")

    df = load_features()
    is_trading_day = check_trading_day()

    # Critical checks
    failures = check_critical(df, is_trading_day)
    if failures:
        for f in failures:
            logger.error(f"CRITICAL: {f}")
        logger.error("Validation failed — halting pipeline")
        sys.exit(1)

    # Warning checks
    warnings = check_warnings(df)
    for w in warnings:
        logger.warning(f"WARNING: {w}")

    # Info
    gpr_days = int(df["gpr_days_stale"].iloc[-1])
    logger.info(f"GPR lag: {gpr_days} days")
    logger.info(f"Quality score: {compute_quality_score(warnings):.2f}")

    # Write quality metadata and save
    df = run_gate(df, failures, warnings)
    save_features(df)

    logger.info("Validation complete — gate passed")