import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    TICKERS,
    SEQ_LEN,
    TECH_COLS,
    RISK_COLS,
    _rsi,
    find_data_dir,
    load_and_split_stock,
    make_sequences,
)
from backtest_swarm import compute_metrics

st.set_page_config(page_title="Quant Trading Swarm — Tomorrow Predictor", layout="wide")


# ==============================================================================
# 1. MODEL LOADING (CACHED)
# ==============================================================================
@st.cache_resource
def load_all_models():
    market_model = tf.keras.models.load_model("models/agent1/lstm_market_agent.keras")
    market_scaler = joblib.load("models/agent1/scaler_market.pkl")

    tokenizer = AutoTokenizer.from_pretrained("models/agent2")
    news_model = AutoModelForSequenceClassification.from_pretrained("models/agent2")
    news_model.eval()

    risk_bundle = joblib.load("models/agent3/risk_logreg.pkl")
    return (
        market_model,
        market_scaler,
        tokenizer,
        news_model,
        risk_bundle["model"],
        risk_bundle["scalers"],
    )


# ==============================================================================
# 2. FEATURE ENGINEERING & LIVE DATA FETCHERS
# ==============================================================================
def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact stationary technical & risk features matching utils.py."""
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[(df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0)].reset_index(drop=True)

    df["Return"] = df["Close"].pct_change()
    df["Return_Lag1"] = df["Return"].shift(1)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    df["Close_MA5_Ratio"] = df["Close"] / df["MA_5"] - 1.0
    df["Close_MA20_Ratio"] = df["Close"] / df["MA_20"] - 1.0
    df["MA5_MA20_Ratio"] = df["MA_5"] / df["MA_20"] - 1.0

    df["RSI_14"] = _rsi(df["Close"]) / 100.0
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Norm"] = macd / df["Close"]
    df["MACD_Hist_Norm"] = (macd - macd_sig) / df["Close"]

    df["Volatility_7d"] = df["Return"].rolling(7).std()
    df["Volatility_21d"] = df["Return"].rolling(21).std()
    df["Daily_Spread"] = (df["High"] - df["Low"]) / df["Close"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


@st.cache_data(ttl=900)
def fetch_live_ohlcv(ticker: str, period: str = "6mo") -> tuple[pd.DataFrame | None, str]:
    """Pull newest daily OHLCV candles from Yahoo Finance."""
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 35:
            return None, "Insufficient live rows returned from Yahoo Finance."
        hist = hist.reset_index()
        cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
        return hist[cols], f"Live Yahoo Finance data up to {pd.to_datetime(hist['Date'].iloc[-1]).strftime('%Y-%m-%d')}"
    except Exception as e:
        return None, f"Live fetch unavailable ({e}). Using local dataset."


@st.cache_data(ttl=900)
def fetch_live_headlines(ticker: str, max_items: int = 5) -> list[str]:
    """Pull latest news headlines for the ticker from Yahoo Finance."""
    try:
        import yfinance as yf

        news_items = yf.Ticker(ticker).news or []
        headlines = []
        for item in news_items:
            title = item.get("title") or item.get("content", {}).get("title")
            if title and isinstance(title, str):
                headlines.append(title.strip())
            if len(headlines) >= max_items:
                break
        return headlines
    except Exception:
        return []


def load_kaggle_sample_headlines(data_dir: str, n: int = 5) -> list[str]:
    """Draw random authentic headlines from local cleaned_news.csv."""
    csv_path = os.path.join(data_dir, "cleaned_news.csv")
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
        if df.shape[1] == 2 and "clean_news" not in df.columns:
            df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=["sentiment", "clean_news"])
        text_col = "clean_news" if "clean_news" in df.columns else max(
            df.columns, key=lambda c: df[c].astype(str).str.len().mean()
        )
        pool = df[text_col].dropna().astype(str).str.strip()
        pool = pool[pool != ""]
        return pool.sample(min(n, len(pool))).tolist()
    except Exception:
        return []


# ==============================================================================
# 3. AGENT INFERENCE & WEIGHTED SWARM CONSENSUS
# ==============================================================================
def predict_headlines_batch(headlines: list[str], tokenizer, news_model) -> tuple[pd.DataFrame, float]:
    """Score one or more headlines with DistilBERT and return per-headline breakdown + mean score."""
    clean = [h.strip() for h in headlines if h and h.strip()]
    if not clean:
        return pd.DataFrame(columns=["Headline", "Sentiment", "Confidence", "Score"]), 0.0

    sentiment_values = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    inputs = tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = news_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    rows = []
    for i, text in enumerate(clean):
        pred_idx = int(torch.argmax(probs[i]).item())
        label = news_model.config.id2label[pred_idx].lower()
        conf = float(probs[i][pred_idx].item())
        score = sentiment_values.get(label, 0.0) * conf
        rows.append({
            "Headline": text,
            "Sentiment": label.upper(),
            "Confidence": f"{conf:.1%}",
            "Score": round(score, 4),
        })

    df_res = pd.DataFrame(rows)
    return df_res, float(df_res["Score"].mean())


def weighted_swarm_decision(
    p_up: float,
    news_score: float,
    p_safe: float,
    w_market: float,
    w_news: float,
    w_risk: float,
    risk_threshold: float = 0.45,
    hard_risk_veto: bool = True,
    buy_cutoff: float = 0.12,
    sell_cutoff: float = -0.12,
) -> dict:
    """
    Combine normalized signals [-1, +1] from all 3 agents using user-defined influence weights.
    - Agent 1 signal: (p_up - 0.5) * 4  -> [-1, +1]
    - Agent 2 signal: news_score         -> [-1, +1]
    - Agent 3 signal: (p_safe - risk_threshold) / max(risk_threshold, 1 - risk_threshold) -> [-1, +1]
    """
    sig_market = float(np.clip((p_up - 0.5) * 4.0, -1.0, 1.0))
    sig_news = float(np.clip(news_score, -1.0, 1.0))
    denom = max(risk_threshold, 1.0 - risk_threshold)
    sig_risk = float(np.clip((p_safe - risk_threshold) / denom, -1.0, 1.0))

    total_w = max(w_market + w_news + w_risk, 1e-6)
    nw_market, nw_news, nw_risk = w_market / total_w, w_news / total_w, w_risk / total_w

    contrib_market = nw_market * sig_market
    contrib_news = nw_news * sig_news
    contrib_risk = nw_risk * sig_risk
    composite_score = contrib_market + contrib_news + contrib_risk

    vetoed = hard_risk_veto and (p_safe <= risk_threshold)

    if vetoed:
        action = "HOLD / CASH (RISK VETO)"
        reason = f"Blocked by Agent 3: Safety {p_safe:.1%} <= threshold {risk_threshold:.0%}"
    elif composite_score >= buy_cutoff:
        action = "BUY / LONG TOMORROW"
        reason = f"Bullish Weighted Consensus ({composite_score:+.3f} >= {buy_cutoff:+.2f})"
    elif composite_score <= sell_cutoff:
        action = "SELL / AVOID TOMORROW"
        reason = f"Bearish Weighted Consensus ({composite_score:+.3f} <= {sell_cutoff:+.2f})"
    else:
        action = "HOLD / NEUTRAL"
        reason = f"Composite Score Neutral ({composite_score:+.3f} within [{sell_cutoff:+.2f}, {buy_cutoff:+.2f}])"

    return {
        "action": action,
        "reason": reason,
        "vetoed": vetoed,
        "composite_score": composite_score,
        "sig_market": sig_market,
        "sig_news": sig_news,
        "sig_risk": sig_risk,
        "contrib_market": contrib_market,
        "contrib_news": contrib_news,
        "contrib_risk": contrib_risk,
        "norm_weights": (nw_market, nw_news, nw_risk),
    }


# ==============================================================================
# 4. VISUAL DIAGRAMS & CHARTS (PLOTLY)
# ==============================================================================
def build_sankey_diagram(ticker: str, res: dict, p_up: float, news_score: float, p_safe: float) -> go.Figure:
    nw_m, nw_n, nw_r = res["norm_weights"]
    labels = [
        f"Market OHLCV ({ticker})",
        "Breaking News Headlines",
        "Volatility & Spread Regime",
        f"Agent 1: LSTM\n(P(Up)={p_up:.1%})",
        f"Agent 2: DistilBERT\n(Score={news_score:+.2f})",
        f"Agent 3: Risk LogReg\n(P(Safe)={p_safe:.1%})",
        f"FINAL SWARM DECISION\n{res['action']}",
    ]
    sources = [0, 1, 2, 3, 4, 5]
    targets = [3, 4, 5, 6, 6, 6]
    values = [
        max(nw_m * 100, 5),
        max(nw_n * 100, 5),
        max(nw_r * 100, 5),
        max(nw_m * 100, 5),
        max(nw_n * 100, 5),
        max(nw_r * 100, 5),
    ]
    final_color = (
        "#10b981"
        if "BUY" in res["action"]
        else ("#ef4444" if "SELL" in res["action"] or "VETO" in res["action"] else "#6b7280")
    )
    node_colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#2563eb", "#7c3aed", "#d97706", final_color]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=22,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(source=sources, target=targets, value=values, color="rgba(148, 163, 184, 0.35)"),
            )
        ]
    )
    fig.update_layout(
        title_text="Swarm Architecture & Weighted Decision Flow",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def build_agent_charts(res: dict) -> tuple[go.Figure, go.Figure]:
    categories = ["Agent 1: Technical (LSTM)", "Agent 2: Sentiment (BERT)", "Agent 3: Safety (Risk)"]
    raw_vals = [
        (res["sig_market"] + 1) * 50,
        (res["sig_news"] + 1) * 50,
        (res["sig_risk"] + 1) * 50,
    ]
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=raw_vals + [raw_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Bullish / Safe Strength (0-100)",
            line_color="#2563eb",
        )
    )
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Agent Signal Profile (50 = Neutral)",
        height=320,
        margin=dict(l=30, r=30, t=45, b=20),
    )

    names = ["Agent 1 (Market)", "Agent 2 (News)", "Agent 3 (Risk)", "Composite Score"]
    vals = [res["contrib_market"], res["contrib_news"], res["contrib_risk"], res["composite_score"]]
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in vals[:3]] + ["#3b82f6"]

    bar = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
        )
    )
    bar.add_hline(y=0, line_dash="dash", line_color="gray")
    bar.update_layout(
        title="Weighted Influence on Final Decision",
        yaxis=dict(range=[-1.05, 1.05], title="Weighted Contribution"),
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return radar, bar


def build_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    sub = df.tail(60).copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{ticker} Price, MA5 & MA20 (Last 60 Trading Days)", "RSI (14)"),
    )
    if all(c in sub.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=sub["Date"],
                open=sub["Open"],
                high=sub["High"],
                low=sub["Low"],
                close=sub["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Close"], name="Close", line=dict(color="#2563eb")), row=1, col=1)

    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MA_5"], name="MA 5", line=dict(color="#f59e0b", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MA_20"], name="MA 20", line=dict(color="#8b5cf6", width=1.4)), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=sub["Date"], y=sub["RSI_14"] * 100, name="RSI 14", line=dict(color="#0ea5e9", width=1.6)),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ==============================================================================
# 5. MAIN STREAMLIT UI
# ==============================================================================
st.title("Multi-Agent Quantitative Trading & Risk Swarm")
st.caption("Next-Day Market Forecasting with Custom CSV Upload, Live Data Pull, Custom Agent Weights & Interactive Visuals")

market_model, market_scaler, tokenizer, news_model, risk_model, risk_scalers = load_all_models()
data_dir = find_data_dir()

# ---------------- SIDEBAR: DATA SOURCE & AGENT WEIGHTS ----------------
st.sidebar.header("1. Asset & Data Source")
data_mode = st.sidebar.radio(
    "Market Data Input",
    [
        "Auto-Pull Latest (Yahoo Finance)",
        "Local Historical Dataset",
        "Upload Custom Market CSV",
        "Append Custom Tomorrow Bar",
    ],
    index=0,
)

uploaded_csv = None
if data_mode == "Upload Custom Market CSV":
    uploaded_csv = st.sidebar.file_uploader(
        "Upload OHLC CSV (min ~42 rows)",
        type=["csv"],
        help="Must include Date, High, Low, Close (and optionally Open, Volume).",
    )
    active_ticker = (
        os.path.splitext(uploaded_csv.name)[0].upper()
        if uploaded_csv is not None
        else "CUSTOM_CSV"
    )
else:
    ticker_choice = st.sidebar.selectbox("Select Portfolio Ticker", TICKERS, index=0)
    custom_ticker = st.sidebar.text_input("Or Enter Any Custom Ticker (Optional)", value="").strip().upper()
    active_ticker = custom_ticker if custom_ticker else ticker_choice

st.sidebar.header("2. Agent Influence Weights")
st.sidebar.caption("Adjust how strongly each agent shapes tomorrow's final decision:")
w_market = st.sidebar.slider("Agent 1 Influence (Market LSTM)", 0.0, 1.0, 0.45, 0.05)
w_news = st.sidebar.slider("Agent 2 Influence (News DistilBERT)", 0.0, 1.0, 0.35, 0.05)
w_risk = st.sidebar.slider("Agent 3 Influence (Risk LogReg)", 0.0, 1.0, 0.20, 0.05)

st.sidebar.header("3. Risk Gate & Thresholds")
risk_threshold = st.sidebar.slider("Agent 3: Min Safety Probability", 0.25, 0.65, 0.45, 0.01)
hard_risk_veto = st.sidebar.checkbox("Enable Hard Risk Veto (Block trades if Unsafe)", value=True)
buy_cutoff = st.sidebar.slider("Composite BUY Threshold", 0.02, 0.40, 0.12, 0.02)

# ---------------- LOAD & PREPARE MARKET DATA ----------------
status_msg = ""
if data_mode == "Upload Custom Market CSV":
    if uploaded_csv is None:
        st.warning("Upload a market `.csv` file in the left sidebar (`Date, Open, High, Low, Close`) to generate tomorrow's prediction.")
        st.stop()
    raw_up = pd.read_csv(uploaded_csv)
    raw_up.columns = [c.strip().title() for c in raw_up.columns]
    if "Close" not in raw_up.columns and "Adj Close" in raw_up.columns:
        raw_up["Close"] = raw_up["Adj Close"]
    required_cols = {"Date", "High", "Low", "Close"}
    if not required_cols.issubset(set(raw_up.columns)):
        st.error(f"CSV is missing required columns: {required_cols - set(raw_up.columns)}")
        st.stop()
    feat_df = engineer_features(raw_up)
    if len(feat_df) < SEQ_LEN:
        st.error(
            f"After computing 21-day indicators, only {len(feat_df)} valid rows remain (need at least {SEQ_LEN}). "
            "Upload a CSV with at least 42 trading days."
        )
        st.stop()
    status_msg = f"Uploaded CSV `{uploaded_csv.name}` ({len(feat_df)} engineered rows up to {feat_df['Date'].iloc[-1].strftime('%Y-%m-%d')})"

elif data_mode == "Auto-Pull Latest (Yahoo Finance)":
    live_raw, status_msg = fetch_live_ohlcv(active_ticker)
    if live_raw is not None:
        feat_df = engineer_features(live_raw)
    else:
        if active_ticker in TICKERS:
            _, _, feat_df = load_and_split_stock(active_ticker, data_dir=data_dir)
        else:
            st.error(f"Could not pull live data for {active_ticker} and it is not in local CSVs.")
            st.stop()
else:
    if active_ticker not in TICKERS:
        st.warning(f"{active_ticker} is not in local CSVs — switching to Auto-Pull Latest.")
        live_raw, status_msg = fetch_live_ohlcv(active_ticker)
        if live_raw is None:
            st.error(status_msg)
            st.stop()
        feat_df = engineer_features(live_raw)
    else:
        _, _, feat_df = load_and_split_stock(active_ticker, data_dir=data_dir)
        status_msg = f"Local dataset up to {feat_df['Date'].iloc[-1].strftime('%Y-%m-%d')}"

# Optional: Append custom today's bar to predict tomorrow
if data_mode == "Append Custom Tomorrow Bar":
    st.sidebar.subheader("Enter Today's OHLC Bar")
    last_close = float(feat_df["Close"].iloc[-1])
    c_open = st.sidebar.number_input("Today Open", value=round(last_close, 2))
    c_high = st.sidebar.number_input("Today High", value=round(last_close * 1.01, 2))
    c_low = st.sidebar.number_input("Today Low", value=round(last_close * 0.99, 2))
    c_close = st.sidebar.number_input("Today Close", value=round(last_close * 1.005, 2))

    raw_cols = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in feat_df.columns]
    base_raw = feat_df[raw_cols].copy()
    new_row = {
        "Date": pd.to_datetime(base_raw["Date"].iloc[-1]) + pd.Timedelta(days=1),
        "Close": c_close,
        "High": max(c_high, c_open, c_close),
        "Low": min(c_low, c_open, c_close),
    }
    if "Open" in raw_cols:
        new_row["Open"] = c_open
    base_raw = pd.concat([base_raw, pd.DataFrame([new_row])], ignore_index=True)
    feat_df = engineer_features(base_raw)
    status_msg = f"Custom bar appended for {new_row['Date'].strftime('%Y-%m-%d')} — predicting tomorrow"

st.info(f"**Active Asset:** `{active_ticker}` | **Data Status:** {status_msg} | **Latest Close:** `${feat_df['Close'].iloc[-1]:.2f}`")

# ---------------- COMPUTE AGENT 1 & AGENT 3 ON LATEST WINDOW ----------------
tech_scaled = feat_df.copy()
tech_scaled[TECH_COLS] = np.clip(market_scaler.transform(tech_scaled[TECH_COLS]), -5, 5)
latest_seq = tech_scaled[TECH_COLS].values[-SEQ_LEN:].reshape(1, SEQ_LEN, len(TECH_COLS)).astype("float32")
p_up_tomorrow = float(market_model.predict(latest_seq, verbose=0)[0, 0])

active_risk_scaler = risk_scalers.get(active_ticker, next(iter(risk_scalers.values())))
latest_risk_x = active_risk_scaler.transform(feat_df[RISK_COLS].iloc[[-1]].values)
p_safe_tomorrow = float(risk_model.predict_proba(latest_risk_x)[0, 1])

# ---------------- HEADLINES INPUT SECTION (AGENT 2) ----------------
st.subheader("1. Breaking News Input (Agent 2 — DistilBERT Sentiment)")

news_source = st.radio(
    "Headline Feed Source",
    ["Auto-Pull Latest Headlines (Yahoo Finance)", "Enter Custom Headlines (One per line)", "Random Sample from Dataset"],
    horizontal=True,
)

if news_source == "Auto-Pull Latest Headlines (Yahoo Finance)":
    pulled = fetch_live_headlines(active_ticker, max_items=5)
    if pulled:
        raw_headlines_text = "\n".join(pulled)
    else:
        st.warning("No live headlines returned right now — enter custom headlines below or switch source.")
        raw_headlines_text = f"{active_ticker} shares steady ahead of quarterly earnings update."
elif news_source == "Random Sample from Dataset":
    if "sample_news_text" not in st.session_state or st.button("Draw New Random Headlines"):
        samples = load_kaggle_sample_headlines(data_dir, n=4)
        st.session_state["sample_news_text"] = (
            "\n".join(samples) if samples else f"{active_ticker} posts strong quarterly revenue growth."
        )
    raw_headlines_text = st.session_state["sample_news_text"]
else:
    raw_headlines_text = (
        f"{active_ticker} beats quarterly earnings expectations and raises guidance.\n"
        f"Analysts upgrade {active_ticker} following strong product demand."
    )

headlines_box = st.text_area(
    "Active Headlines Evaluated for Tomorrow's Forecast (Edit or Add Lines):",
    value=raw_headlines_text,
    height=105,
)
headline_list = [line.strip() for line in headlines_box.splitlines() if line.strip()]
news_df, mean_news_score = predict_headlines_batch(headline_list, tokenizer, news_model)

# ---------------- COMPUTE FINAL SWARM DECISION ----------------
swarm_res = weighted_swarm_decision(
    p_up=p_up_tomorrow,
    news_score=mean_news_score,
    p_safe=p_safe_tomorrow,
    w_market=w_market,
    w_news=w_news,
    w_risk=w_risk,
    risk_threshold=risk_threshold,
    hard_risk_veto=hard_risk_veto,
    buy_cutoff=buy_cutoff,
    sell_cutoff=-buy_cutoff,
)

# ---------------- DISPLAY FINAL VERDICT BANNER + INDIVIDUAL AGENT CARDS ----------------
st.subheader("2. Tomorrow's Final Swarm Forecast & Individual Agent Votes")

action_text = swarm_res["action"]
comp_score = swarm_res["composite_score"]

if "BUY" in action_text:
    banner_bg = "linear-gradient(135deg, #065f46 0%, #10b981 100%)"
    border_col = "#34d399"
    badge_icon = "🚀"
    sub_label = "BULLISH CONSENSUS — ENTER / HOLD LONG POSITION TOMORROW"
elif "SELL" in action_text:
    banner_bg = "linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%)"
    border_col = "#f87171"
    badge_icon = "📉"
    sub_label = "BEARISH CONSENSUS — EXIT / SHORT-AVOID TOMORROW"
elif "VETO" in action_text:
    banner_bg = "linear-gradient(135deg, #78350f 0%, #d97706 100%)"
    border_col = "#fbbf24"
    badge_icon = "🛡️"
    sub_label = "RISK CIRCUIT BREAKER ACTIVE — STAY IN CASH TOMORROW"
else:
    banner_bg = "linear-gradient(135deg, #1e293b 0%, #475569 100%)"
    border_col = "#94a3b8"
    badge_icon = "⏸️"
    sub_label = "MIXED / NEUTRAL SIGNALS — HOLD CASH & WAIT FOR CONFIRMATION"

# Map composite score [-1, +1] to percentage [0%, 100%] for the visual needle
gauge_pct = int(np.clip((comp_score + 1.0) * 50.0, 2, 98))

st.markdown(
    f"""
    <div style="
        background: {banner_bg};
        border: 2px solid {border_col};
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        color: #ffffff;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 1.5px; opacity: 0.9; text-transform: uppercase;">
                    FINAL SWARM VERDICT FOR {active_ticker} (TOMORROW)
                </div>
                <div style="font-size: 2.35rem; font-weight: 900; letter-spacing: 0.5px; margin: 4px 0;">
                    {badge_icon} {action_text}
                </div>
                <div style="font-size: 1.0rem; font-weight: 500; opacity: 0.95;">
                    {sub_label}
                </div>
            </div>
            <div style="
                background: rgba(0, 0, 0, 0.28);
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 12px;
                padding: 14px 22px;
                text-align: center;
                min-width: 170px;
            ">
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85;">
                    Composite Score
                </div>
                <div style="font-size: 2.0rem; font-weight: 800;">
                    {comp_score:+.3f}
                </div>
                <div style="font-size: 0.78rem; opacity: 0.85;">
                    Range: -1.00 to +1.00
                </div>
            </div>
        </div>
        <div style="margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; opacity: 0.85; margin-bottom: 4px;">
                <span>STRONG SELL (-1.0)</span>
                <span>NEUTRAL (0.0)</span>
                <span>STRONG BUY (+1.0)</span>
            </div>
            <div style="position: relative; height: 10px; background: rgba(255,255,255,0.22); border-radius: 999px;">
                <div style="
                    position: absolute;
                    left: {gauge_pct}%;
                    top: -4px;
                    transform: translateX(-50%);
                    width: 18px;
                    height: 18px;
                    background: #ffffff;
                    border: 3px solid #0f172a;
                    border-radius: 50%;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                "></div>
            </div>
        </div>
        <div style="margin-top: 12px; font-size: 0.9rem; opacity: 0.92;">
            <strong>Trigger Rule:</strong> {swarm_res['reason']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Individual Agent Cards directly underneath the Hero Banner
nw_m, nw_n, nw_r = swarm_res["norm_weights"]
col1, col2, col3 = st.columns(3)

with col1:
    a1_dir = "BULLISH (UP)" if p_up_tomorrow > 0.52 else ("BEARISH (DOWN)" if p_up_tomorrow < 0.48 else "NEUTRAL")
    with st.container(border=True):
        st.metric(
            label=f"Agent 1: Market LSTM ({nw_m:.0%} influence)",
            value=a1_dir,
            delta=f"P(Up): {p_up_tomorrow:.1%} | Weighted: {swarm_res['contrib_market']:+.3f}",
        )

with col2:
    a2_dir = "BULLISH" if mean_news_score > 0.15 else ("BEARISH" if mean_news_score < -0.15 else "NEUTRAL")
    with st.container(border=True):
        st.metric(
            label=f"Agent 2: News DistilBERT ({nw_n:.0%} influence)",
            value=a2_dir,
            delta=f"Score: {mean_news_score:+.3f} | Weighted: {swarm_res['contrib_news']:+.3f}",
        )

with col3:
    a3_dir = "SAFE REGIME" if p_safe_tomorrow > risk_threshold else "CRASH RISK (VETO)"
    with st.container(border=True):
        st.metric(
            label=f"Agent 3: Risk LogReg ({nw_r:.0%} influence)",
            value=a3_dir,
            delta=f"P(Safe): {p_safe_tomorrow:.1%} | Weighted: {swarm_res['contrib_risk']:+.3f}",
            delta_color="normal" if p_safe_tomorrow > risk_threshold else "inverse",
        )
    if not news_df.empty:
        st.markdown("#### Individual Headline Ratings (DistilBERT)")
        st.dataframe(news_df, use_container_width=True, hide_index=True)
# ---------------- DIAGRAMS & CHARTS ----------------
st.subheader("3. Swarm Architecture, Influence Diagrams & Technical Charts")

st.plotly_chart(
    build_sankey_diagram(active_ticker, swarm_res, p_up_tomorrow, mean_news_score, p_safe_tomorrow),
    use_container_width=True,
)

c_left, c_right = st.columns(2)
radar_fig, bar_fig = build_agent_charts(swarm_res)
with c_left:
    st.plotly_chart(radar_fig, use_container_width=True)
with c_right:
    st.plotly_chart(bar_fig, use_container_width=True)

st.plotly_chart(build_candlestick_chart(feat_df, active_ticker), use_container_width=True)

# ---------------- HISTORICAL TEST-SPLIT BENCHMARK ----------------
if active_ticker in TICKERS and data_mode != "Upload Custom Market CSV":
    with st.expander(f"4. Historical Out-of-Sample Backtest Benchmark ({active_ticker} Test Split)"):
        _, _, test_df = load_and_split_stock(active_ticker, data_dir=data_dir)
        t_scaled = test_df.copy()
        t_scaled[TECH_COLS] = np.clip(market_scaler.transform(t_scaled[TECH_COLS]), -5, 5)
        X_seq_hist, _ = make_sequences(t_scaled, TECH_COLS, "Target_Up", seq_len=SEQ_LEN)
        p_up_hist = market_model.predict(X_seq_hist, verbose=0).ravel()

        aligned_hist = test_df.iloc[SEQ_LEN - 1 :].reset_index(drop=True)
        X_risk_hist = risk_scalers[active_ticker].transform(aligned_hist[RISK_COLS].values)
        p_safe_hist = risk_model.predict_proba(X_risk_hist)[:, 1]

        next_rets = aligned_hist["Next_Return"].values
        swarm_pos = np.zeros(len(aligned_hist))
        tech_pos = (p_up_hist > 0.52).astype(float)

        for i in range(len(aligned_hist)):
            d = weighted_swarm_decision(
                p_up=p_up_hist[i],
                news_score=0.0,
                p_safe=p_safe_hist[i],
                w_market=w_market,
                w_news=w_news,
                w_risk=w_risk,
                risk_threshold=risk_threshold,
                hard_risk_veto=hard_risk_veto,
                buy_cutoff=buy_cutoff,
                sell_cutoff=-buy_cutoff,
            )
            swarm_pos[i] = 1.0 if "BUY" in d["action"] else 0.0

        cost = 5.0 / 10000.0
        s_ret = swarm_pos * next_rets - np.abs(np.diff(swarm_pos, prepend=0.0)) * cost
        t_ret = tech_pos * next_rets - np.abs(np.diff(tech_pos, prepend=0.0)) * cost

        m_df = pd.DataFrame([
            {"Strategy": "Weighted Swarm", **compute_metrics(s_ret), "Time_In_Market_%": round(swarm_pos.mean() * 100, 1)},
            {"Strategy": "Agent 1 (LSTM Only)", **compute_metrics(t_ret), "Time_In_Market_%": round(tech_pos.mean() * 100, 1)},
            {"Strategy": "Buy & Hold", **compute_metrics(next_rets), "Time_In_Market_%": 100.0},
        ])
        st.dataframe(m_df, use_container_width=True, hide_index=True)

        eq_fig = go.Figure()
        eq_fig.add_trace(
            go.Scatter(x=aligned_hist["Date"], y=np.cumprod(1 + s_ret), name="Weighted Swarm", line=dict(color="#10b981", width=2.2))
        )
        eq_fig.add_trace(
            go.Scatter(x=aligned_hist["Date"], y=np.cumprod(1 + t_ret), name="Agent 1 Only", line=dict(color="#3b82f6", width=1.6))
        )
        eq_fig.add_trace(
            go.Scatter(x=aligned_hist["Date"], y=np.cumprod(1 + next_rets), name="Buy & Hold", line=dict(color="#94a3b8", width=1.4))
        )
        eq_fig.update_layout(
            title=f"{active_ticker} Cumulative Equity Growth (Test Split)",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(eq_fig, use_container_width=True)