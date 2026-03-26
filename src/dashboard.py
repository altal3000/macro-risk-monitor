import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
import ast

DB_PATH = "macro_risk_monitor.duckdb"

FEATURE_LABELS = {
    "brent_ret_1d":        "Brent crude daily move",
    "brent_ret_5d":        "Brent crude weekly move",
    "wti_ret_1d":          "WTI crude daily move",
    "brent_wti_spread":    "Brent-WTI spread widening",
    "brent_vol_20d":       "Oil price volatility",
    "ng_ret_1d":           "Natural gas price move",
    "xle_ret_1d":          "Energy equities move",
    "xle_brent_ratio":     "Equities-to-oil divergence",
    "ovx_zscore":          "Oil volatility spike",
    "vix_zscore":          "Broad market fear spike",
    "yield_curve":         "Yield curve inversion",
    "yield_curve_chg_20d": "Yield curve rapid shift",
    "dxy_zscore":          "Dollar strength spike",
    "hy_spread":           "Credit market stress",
    "hy_spread_chg_5d":    "Credit stress acceleration",
    "fed_rate":            "Elevated interest rates",
    "cpi_yoy":             "Inflation pressure",
    "geo_signal":          "Geopolitical risk",
    "gpr_oil_zscore":      "Oil-specific geopolitical risk",
    "gpr_is_stale":        "Geopolitical data delay",
    "gpr_days_stale":      "Geopolitical data lag"
}

st.set_page_config(page_title="Macro Risk Monitor", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    hr {
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #444 !important;
        border-color: #444 !important;
        color: white !important;
        font-size: 0.8rem !important;
    }
    button[kind="secondary"] {
        background-color: white !important;
        border-color: #ccc !important;
        color: #444 !important;
        font-size: 0.8rem !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    import boto3
    s3 = boto3.client("s3")
    s3.download_file("macro-risk-monitor", "macro_risk_monitor.duckdb", "/tmp/macro_risk_monitor.duckdb")
    con = duckdb.connect("/tmp/macro_risk_monitor.duckdb", read_only=True)
    df = con.execute("SELECT * FROM risk_scores ORDER BY date").fetchdf()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()
latest = df.iloc[-1]

st.title("Macro Risk Monitor")
st.caption("Daily anomaly detection for energy markets")
last_date = df["date"].max().strftime("%B %d, %Y")
st.markdown(
    f"<div style='color:#999;font-size:0.85rem;margin-top:-8px'>Last analysed: {last_date}</div>",
    unsafe_allow_html=True
)

st.divider()

# --- Energy Market Stress Index ---
st.subheader("Energy Market Stress Index")

def get_bar_colour(pct: float) -> str:
    if pct >= 95:   return "#d32f2f"
    elif pct >= 80: return "#f57c00"
    elif pct >= 60: return "#f9a825"
    else:           return "#2e7d32"

def get_tab_emoji(pct: float) -> str:
    if pct >= 95:   return "🔴"
    elif pct >= 80: return "🟠"
    elif pct >= 60: return "🟡"
    else:           return "🟢"

score = float(latest["risk_score"])

windows = {
    "Quarter":    (df[df["date"] >= df["date"].max() - pd.DateOffset(months=3)]["risk_score"], bool(latest["anomaly_qtr"])),
    "1-year":     (df[df["date"] >= df["date"].max() - pd.DateOffset(years=1)]["risk_score"],  bool(latest["anomaly_1y"])),
    "5-year":     (df[df["date"] >= df["date"].max() - pd.DateOffset(years=5)]["risk_score"],  bool(latest["anomaly_5y"])),
    "Historical": (df["risk_score"], bool(latest["anomaly_static"])),
}

pcts = {
    label: round((series < score).mean() * 100, 1)
    for label, (series, _) in windows.items()
}

anomalous_windows = [label for label, (_, flag) in windows.items() if flag]
if anomalous_windows:
    st.error(f"⚠️ Anomaly detected — {', '.join(anomalous_windows)} indicator{'s' if len(anomalous_windows) > 1 else ''} above threshold")

tab_labels = [f"{get_tab_emoji(pcts[label])} {label}" for label in windows]

col_gauge, col_drivers = st.columns([1, 1])

with col_gauge:
    tabs = st.tabs(tab_labels)
    for tab, (label, (series, is_anomaly)) in zip(tabs, windows.items()):
        with tab:
            pct = pcts[label]
            bar_colour = get_bar_colour(pct)
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=pct,
                gauge={
                    "axis": {"range": [0, 100]},
                    "threshold": {"line": {"color": "#d32f2f", "width": 2}, "thickness": 0.75, "value": 95},
                    "bar": {"color": bar_colour},
                    "bgcolor": "white",
                    "steps": []
                },
                title={"text": f"vs. {label}", "font": {"size": 13, "color": "#666"}}
            ))
            fig.add_annotation(
                x=0.5, y=0.0,
                text=f"<b>{pct}%</b>",
                showarrow=False,
                font=dict(size=40, color=bar_colour),
                xref="paper", yref="paper"
            )
            fig.update_layout(
                height=220,
                margin=dict(t=30, b=20, l=60, r=60),
                paper_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center;margin-top:-10px'>"
                f"<div style='color:#666;font-size:0.85rem'>Higher market stress than {pct}% of days in this window</div>"
                f"<div style='color:#999;font-size:0.75rem;margin-top:2px'>Anomaly score: {score:.2f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

with col_drivers:
    st.markdown("**Top drivers today**")
    drivers = ast.literal_eval(latest["top_drivers"])
    values  = ast.literal_eval(latest["top_driver_values"])
    total   = sum(values[d] for d in drivers)
    top4    = drivers[:4]
    top4_pcts   = {d: round(values[d] / total * 100, 1) for d in top4}
    others_pct  = round(100 - sum(top4_pcts.values()), 1)
    labels      = [FEATURE_LABELS.get(d, d) for d in top4] + ["Others"]
    pct_values  = list(top4_pcts.values()) + [others_pct]
    drivers_df  = pd.DataFrame({"feature": labels, "importance": pct_values}).sort_values("importance", ascending=True)

    fig_drivers = go.Figure(go.Bar(
        x=drivers_df["importance"], y=drivers_df["feature"],
        orientation="h", marker_color="#666"
    ))
    fig_drivers.update_layout(
        height=260, margin=dict(t=10, b=10, l=0, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="% of total explanation", ticksuffix="%"),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_drivers, use_container_width=True)
    st.markdown(
        "<div style='color:#999;font-size:0.75rem;margin-top:-10px'>"
        f"Share of today's anomaly score explained at 75% by each signal"
        "</div>",
        unsafe_allow_html=True
    )

st.divider()

# --- Stress history ---
st.subheader("Stress history")

WINDOWS_TS = {"3m": 90, "1y": 365, "5y": 1825, "All": None}

if "ts_window" not in st.session_state:
    st.session_state.ts_window = "1y"

# Handle clicks before render
for label in WINDOWS_TS:
    if st.session_state.get(f"btn_{label}"):
        st.session_state.ts_window = label

btn_cols = st.columns([1, 1, 1, 1, 8])
for col, label in zip(btn_cols, WINDOWS_TS):
    with col:
        st.button(
            label,
            key=f"btn_{label}",
            use_container_width=True,
            type="primary" if st.session_state.ts_window == label else "secondary"
        )

days = WINDOWS_TS[st.session_state.ts_window]
min_date = df["date"].min().date()
max_date = df["date"].max().date()

if days:
    end_date = st.slider(
        "end_date",
        min_value=min_date + pd.Timedelta(days=days),
        max_value=max_date,
        value=max_date,
        format="YYYY-MM-DD",
        label_visibility="collapsed"
    )
    start_date = end_date - pd.Timedelta(days=days)
else:
    start_date = min_date
    end_date   = max_date

df_ts = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df_ts["date"], y=df_ts["risk_score"],
    mode="lines", name="Stress score",
    line=dict(color="#444", width=1.2)
))
fig_ts.add_trace(go.Scatter(
    x=df_ts[df_ts["anomaly_static"]]["date"],
    y=df_ts[df_ts["anomaly_static"]]["risk_score"],
    mode="markers", name="Anomaly",
    marker=dict(color="#d32f2f", size=5)
))
fig_ts.update_layout(
    height=300, margin=dict(t=10, b=10, l=0, r=20),
    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Stress score")
)
st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# --- Recent anomalies ---

with st.expander("Anomaly log", expanded=False):
    log_df = df[df["anomaly_static"]].sort_values("date", ascending=False).copy()

    def get_top_drivers_str(row):
        try:
            drivers = ast.literal_eval(row["top_drivers"])
            values  = ast.literal_eval(row["top_driver_values"])
            total   = sum(values[d] for d in drivers)
            top3    = drivers[:3]
            parts   = [f"{FEATURE_LABELS.get(d, d)} ({round(values[d]/total*100)}%)" for d in top3]
            return " · ".join(parts)
        except:
            return ""

    log_df["Date"]         = log_df["date"].dt.strftime("%Y-%m-%d")
    log_df["Stress score"] = log_df["risk_score"].round(2).astype(str)
    log_df["Top drivers"]  = log_df.apply(get_top_drivers_str, axis=1)

    st.dataframe(
        log_df[["Date", "Stress score", "Top drivers"]],
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Date":         st.column_config.TextColumn("Date",         width="small"),
            "Stress score": st.column_config.TextColumn("Anomaly score", width="small"),
            "Top drivers":  st.column_config.TextColumn("Top drivers",  width="large"),
        }
    )
    st.markdown(
        "<div style='color:#999;font-size:0.75rem;margin-top:4px'>"
        "Anomaly scores above 1.5 indicate historically anomalous conditions. "
        "The scale has no upper bound — the highest recorded value was 3.2 during the COVID-19 oil crash in March 2020."
        "</div>",
        unsafe_allow_html=True
    )

# --- Methodology ---

with st.expander("Methodology", expanded=False):
    st.markdown("""
The Macro Risk Monitor analyses daily energy market conditions across 21 signals spanning 
oil and gas prices, equity markets, macroeconomic indicators, and geopolitical risk. 
Each day, the system produces an anomaly score reflecting how unusual current conditions 
are relative to history. Scores above 1.5 indicate historically anomalous conditions. 
The top drivers panel identifies which signals are contributing most to today's score, 
covering 75% of the total explanation. The percentile gauges show how today's score ranks 
relative to four time windows, each grounded in a distinct policy cycle: the current quarter, 
the past year, a five-year geopolitical regime window, and the full 18-year history.
    """)