import os
import logging
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
import duckdb
import mlflow
import mlflow.sklearn
import shap
from sklearn.ensemble import IsolationForest
import ruptures as rpt
import exchange_calendars as xcals
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

S3_BUCKET = "macro-risk-monitor"
s3 = boto3.client("s3")
TODAY = datetime.today().strftime("%Y-%m-%d")
DB_PATH = "macro_risk_monitor.duckdb"

def get_last_trading_day() -> str:
    cal = xcals.get_calendar("XNYS")
    check = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    while not cal.is_session(check):
        check -= pd.Timedelta(days=1)
    return check.strftime("%Y-%m-%d")

LAST_TRADING_DAY = get_last_trading_day()

PRICE_BLOCK  = ["brent_ret_1d","brent_ret_5d","wti_ret_1d","brent_wti_spread",
                "brent_vol_20d","ng_ret_1d","xle_ret_1d","xle_brent_ratio",
                "ovx_zscore","vix_zscore"]
MACRO_BLOCK  = ["yield_curve","yield_curve_chg_20d","dxy_zscore","hy_spread",
                "hy_spread_chg_5d","fed_rate","cpi_yoy"]
GEO_BLOCK    = ["geo_signal","gpr_oil_zscore","gpr_is_stale","gpr_days_stale"]
ALL_FEATURES = PRICE_BLOCK + MACRO_BLOCK + GEO_BLOCK

ROLLING_WINDOWS = {
    "5y":  1260,
    "1y":  252,
    "qtr": 63
}
STATIC_THRESHOLD = 1.53
ROLLING_PERCENTILE = 0.95

# --- Load features ---
def load_features() -> pd.DataFrame:
    s3.download_file(S3_BUCKET, "features/features_window.parquet", "/tmp/features_window.parquet")
    df = pd.read_parquet("/tmp/features_window.parquet")
    df.index = pd.to_datetime(df.index)
    logger.info(f"Loaded features: {df.shape}")
    return df

# --- Reduce features ---
def reduce_features(df: pd.DataFrame, threshold: float = 0.85) -> list:
    corr = df[ALL_FEATURES].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    reduced = [f for f in ALL_FEATURES if f not in to_drop]
    logger.info(f"Reduced features: {len(ALL_FEATURES)} → {len(reduced)} (dropped: {to_drop})")
    return reduced

# --- Isolation Forest ---
def fit_isolation_forest(df: pd.DataFrame, features: list) -> tuple:
    weights = df["data_quality_score"].values
    X = df[features].values

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X, sample_weight=weights)

    scores = -iso.decision_function(X)
    scores_norm = (scores - scores.mean()) / scores.std()
    scores_norm = scores_norm - scores_norm.min()

    logger.info(f"IsolationForest fitted on {len(features)} features, "
                f"mean={scores_norm.mean():.3f}, std={scores_norm.std():.3f}")
    return iso, pd.Series(scores_norm, index=df.index)

# --- Z-score ---
def compute_zscore_signal(df: pd.DataFrame, window: int = 60) -> pd.Series:

    def rolling_zscore(series):
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std().clip(lower=1e-6)
        return (series - mu) / sigma

    zscores = df[ALL_FEATURES].apply(rolling_zscore)

    scores_price = zscores[PRICE_BLOCK].abs().max(axis=1)
    scores_macro = zscores[MACRO_BLOCK].abs().max(axis=1)
    scores_geo   = zscores[GEO_BLOCK].abs().max(axis=1)

    combined = (scores_price + scores_macro + scores_geo) / 3
    scores_norm = combined.clip(upper=5) / 5

    logger.info("Z-score signal computed")
    return scores_norm

# --- Change point ---
def compute_changepoint_signal(df: pd.DataFrame, penalty: int = 10) -> pd.Series:
    scores = pd.Series(0.0, index=df.index)
    weekly = df[ALL_FEATURES].resample("W").last().dropna()

    for col in ALL_FEATURES:
        series = weekly[col].values
        try:
            algo = rpt.Pelt(model="rbf").fit(series)
            breakpoints = algo.predict(pen=penalty)
            for bp in breakpoints[:-1]:
                if bp < len(weekly):
                    weekly_date = weekly.index[bp]
                    scores.loc[weekly_date] += 1
        except Exception:
            continue

    scores_norm = scores / len(ALL_FEATURES)
    logger.info("Change point signal computed")
    return scores_norm

# --- Ensemble ---
def ensemble(
    scores_if: pd.Series,
    scores_z: pd.Series,
    scores_cp: pd.Series,
    w_if: float = 0.4,
    w_z: float = 0.35,
    w_cp: float = 0.25
) -> pd.Series:

    idx = scores_if.index
    scores_z  = scores_z.reindex(idx).fillna(0)
    scores_cp = scores_cp.reindex(idx).fillna(0)

    risk_score = w_if * scores_if + w_z * scores_z + w_cp * scores_cp
    logger.info(f"Ensemble: mean={risk_score.mean():.3f}")
    return risk_score

# --- Compute anomaly flags ---
def compute_anomaly_flags(risk_score: pd.Series) -> pd.DataFrame:
    flags = pd.DataFrame(index=risk_score.index)

    flags["anomaly_static"] = risk_score > STATIC_THRESHOLD

    for name, window in ROLLING_WINDOWS.items():
        rolling_threshold = risk_score.rolling(window, min_periods=window//2).quantile(ROLLING_PERCENTILE)
        flags[f"anomaly_{name}"] = risk_score > rolling_threshold

    logger.info(f"Anomaly rates — static: {flags['anomaly_static'].mean():.3f}, "
                f"5y: {flags['anomaly_5y'].mean():.3f}, "
                f"1y: {flags['anomaly_1y'].mean():.3f}, "
                f"qtr: {flags['anomaly_qtr'].mean():.3f}")
    return flags

# --- SHAP + z-score attribution ---
def compute_shap(iso: IsolationForest, df: pd.DataFrame, features: list,
                 scores_z_raw: pd.DataFrame) -> pd.DataFrame:

    explainer = shap.TreeExplainer(iso)
    shap_values = np.abs(explainer.shap_values(df[features].values))
    shap_df = pd.DataFrame(shap_values, index=df.index, columns=features)
    shap_norm = shap_df.div(shap_df.sum(axis=1), axis=0) * 0.4

    def rolling_zscore(series):
        mu = series.rolling(60).mean()
        sigma = series.rolling(60).std().clip(lower=1e-6)
        return (series - mu) / sigma

    z_abs = scores_z_raw.apply(rolling_zscore).abs()
    z_norm = z_abs.div(z_abs.sum(axis=1), axis=0).fillna(0) * 0.35
    z_norm = z_norm.reindex(columns=features, fill_value=0)

    combined = shap_norm.add(z_norm, fill_value=0)

    today_top = combined.iloc[-1].sort_values(ascending=False)
    logger.info(f"Top drivers today (75% attribution): {today_top.head(3).index.tolist()}")
    return combined

# --- DuckDB setup ---
def setup_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS risk_scores (
            date                DATE PRIMARY KEY,
            risk_score          FLOAT,
            anomaly_static      BOOLEAN,
            anomaly_5y          BOOLEAN,
            anomaly_1y          BOOLEAN,
            anomaly_qtr         BOOLEAN,
            scores_if           FLOAT,
            scores_z            FLOAT,
            scores_cp           FLOAT,
            top_drivers         VARCHAR,
            top_driver_values   VARCHAR,
            data_quality_flags  VARCHAR,
            gpr_is_stale        BOOLEAN,
            model_version       VARCHAR
        )
    """)
    logger.info("DuckDB ready")
    return con

# --- Write record ---
def write_record(con: duckdb.DuckDBPyConnection, date: str, risk_score: float,
                 anomaly_static: bool, anomaly_5y: bool, anomaly_1y: bool,
                 anomaly_qtr: bool, scores_if: float, scores_z: float,
                 scores_cp: float, top_drivers: pd.Series,
                 quality_flags: str, gpr_is_stale: bool, model_version: str) -> None:

    drivers_list   = top_drivers.sort_values(ascending=False).head(5).index.tolist()
    drivers_values = top_drivers.sort_values(ascending=False).head(5).to_dict()

    con.execute("""
        INSERT OR REPLACE INTO risk_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        date,
        float(risk_score),
        bool(anomaly_static),
        bool(anomaly_5y),
        bool(anomaly_1y),
        bool(anomaly_qtr),
        float(scores_if),
        float(scores_z),
        float(scores_cp),
        str(drivers_list),
        str(drivers_values),
        quality_flags,
        bool(gpr_is_stale),
        model_version
    ])

# --- Train and score ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Score all historical rows")
    args = parser.parse_args()

    logger.info(f"Starting train/score — {TODAY}, scoring for {LAST_TRADING_DAY}")

    df_full  = load_features()                          # full window including NaN rows
    df_train = df_full.dropna(subset=ALL_FEATURES)      # clean rows for model training

    quality_flags = df_full["data_quality_flags"].iloc[-1]
    gpr_is_stale  = bool(df_full["gpr_is_stale"].iloc[-1])

    with mlflow.start_run():
        reduced_features = reduce_features(df_train)

        iso, scores_if = fit_isolation_forest(df_train, reduced_features)
        scores_z       = compute_zscore_signal(df_train)
        scores_cp      = compute_changepoint_signal(df_train)

        risk_score = ensemble(scores_if, scores_z, scores_cp)
        flags      = compute_anomaly_flags(risk_score)
        shap_df    = compute_shap(iso, df_train, reduced_features, df_train[reduced_features])

        mlflow.log_param("features_if", reduced_features)
        mlflow.log_param("contamination", 0.05)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("static_threshold", STATIC_THRESHOLD)
        mlflow.log_param("rolling_percentile", ROLLING_PERCENTILE)
        mlflow.log_metric("mean_risk_score", float(risk_score.mean()))
        mlflow.log_metric("anomaly_rate_static", float(flags["anomaly_static"].mean()))
        mlflow.log_metric("anomaly_rate_1y", float(flags["anomaly_1y"].mean()))
        run_id = mlflow.active_run().info.run_id

        con = setup_duckdb()

        if args.backfill:
            logger.info("Backfilling all historical scores...")
            for date in df_train.index:
                date_str = date.strftime("%Y-%m-%d")
                write_record(
                    con=con,
                    date=date_str,
                    risk_score=float(risk_score.loc[date]),
                    anomaly_static=bool(flags.loc[date, "anomaly_static"]),
                    anomaly_5y=bool(flags.loc[date, "anomaly_5y"]),
                    anomaly_1y=bool(flags.loc[date, "anomaly_1y"]),
                    anomaly_qtr=bool(flags.loc[date, "anomaly_qtr"]),
                    scores_if=float(scores_if.loc[date]),
                    scores_z=float(scores_z.loc[date]),
                    scores_cp=float(scores_cp.loc[date]),
                    top_drivers=shap_df.loc[date],
                    quality_flags=str(df_full.loc[date, "data_quality_flags"]),
                    gpr_is_stale=bool(df_full.loc[date, "gpr_is_stale"]),
                    model_version=run_id
                )
            logger.info(f"Backfill complete — {len(df_train)} rows written")
        else:
            # Find the score for LAST_TRADING_DAY
            # If it was dropped due to NaN, use the last available scored date
            score_date = pd.Timestamp(LAST_TRADING_DAY)
            if score_date not in risk_score.index:
                score_date = risk_score.index[-1]
                logger.warning(f"LAST_TRADING_DAY {LAST_TRADING_DAY} not in scored data — using {score_date.date()}")

            write_record(
                con=con,
                date=score_date.strftime("%Y-%m-%d"),
                risk_score=float(risk_score.loc[score_date]),
                anomaly_static=bool(flags.loc[score_date, "anomaly_static"]),
                anomaly_5y=bool(flags.loc[score_date, "anomaly_5y"]),
                anomaly_1y=bool(flags.loc[score_date, "anomaly_1y"]),
                anomaly_qtr=bool(flags.loc[score_date, "anomaly_qtr"]),
                scores_if=float(scores_if.loc[score_date]),
                scores_z=float(scores_z.loc[score_date]),
                scores_cp=float(scores_cp.loc[score_date]),
                top_drivers=shap_df.loc[score_date],
                quality_flags=quality_flags,
                gpr_is_stale=gpr_is_stale,
                model_version=run_id
            )
            logger.info(f"Written score for {score_date.date()}")

        con.close()

        # Upload DuckDB to S3
        s3.upload_file(DB_PATH, S3_BUCKET, "macro_risk_monitor.duckdb")
        logger.info("DuckDB uploaded to S3")

    logger.info("Train/score complete")