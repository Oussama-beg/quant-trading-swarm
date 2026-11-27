import os
import pickle
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None


# ---------------------------------------------------------------------------
# Page & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantitative Trading Swarm — Live & Uploaded Data Decisions",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .decision-card {
        background: #141820;
        border: 1px solid #262c3a;
        border-radius: 10px;
        padding: 16px 18px;
    }
    .card-label {
        font-size: 0.85rem;
        color: #9aa4b2;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f3f6fa;
        line-height: 1.15;
        margin-bottom: 10px;
    }
    .card-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        background: #173326;
        color: #4ade80;
        border: 1px solid #24543c;
    }
    .card-badge-down {
        background: #33191d;
        color: #f87171;
        border-color: #5c262e;
    }
    .card-badge-neutral {
        background: #272b35;
        color: #cbd5e1;
        border-color: #3d4454;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Feature Engineering & Model Loaders
# ---------------------------------------------------------------------------
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True)
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out["Volume"] = out["Volume"].fillna(0.0)
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    close = out["Close"]
    out["MA_5"] = close.rolling(5, min_periods=1).mean()
    out["MA_20"] = close.rolling(20, min_periods=1).mean()
    out["Return_1d"] = close.pct_change().fillna(0.0)
    out["Return_5d"] = close.pct_change(5).fillna(0.0)

    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["RSI_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    return out


@st.cache_resource(show_spinner=False)
def load_lstm_agent():
    model_path = "models/agent1/lstm_market_agent.keras"
    scaler_path = "models/agent1/scaler_market.pkl"
    model = None
    scaler = None
    if keras_load_model is not None and os.path.exists(model_path):
        try:
            model = keras_load_model(model_path)
        except Exception:
            model = None
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None
    return model, scaler


@st.cache_resource(show_spinner=False)
def load_sentiment_agent():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        return None, None, None
    candidates = [
        "models/agent2",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
    for path in candidates:
        if os.path.exists(path) or path.startswith("distilbert-"):
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                return tokenizer, model, device
            except Exception:
                continue
    return None, None, None


@st.cache_resource(show_spinner=False)
def load_combiner_model():
    path = "models/agent3/combiner_lr.pkl"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_agent1(feat_df: pd.DataFrame) -> float:
    if feat_df.empty:
        return 0.5
    model, scaler = load_lstm_agent()
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_20", "Return_1d", "Return_5d", "RSI_14", "MACD"]
    avail_cols = [c for c in feature_cols if c in feat_df.columns]

    if model is not None and scaler is not None and len(feat_df) >= 20 and len(avail_cols) == len(feature_cols):
        try:
            window = feat_df[feature_cols].tail(20).to_numpy(dtype=np.float32)
            scaled = scaler.transform(window).reshape(1, 20, len(feature_cols))
            pred = float(np.squeeze(model.predict(scaled, verbose=0)))
            return float(np.clip(pred, 0.0, 1.0))
        except Exception:
            pass

    # Fallback technical score when Keras/scaler artifacts are absent
    last = feat_df.iloc[-1]
    r1 = float(np.clip(last.get("Return_1d", 0.0) * 12.0, -0.25, 0.25))
    r5 = float(np.clip(last.get("Return_5d", 0.0) * 5.0, -0.25, 0.25))
    rsi = float(last.get("RSI_14", 50.0))
    rsi_tilt = (50.0 - rsi) / 250.0 if rsi < 35.0 else (rsi - 50.0) / 300.0
    prob = 0.50 + r1 + r5 + float(np.clip(rsi_tilt, -0.15, 0.15))
    return float(np.clip(prob, 0.01, 0.99))


def score_agent2(headlines: List[Tuple[str, str]]) -> Tuple[float, pd.DataFrame]:
    if not headlines:
        return 0.0, pd.DataFrame(columns=["date", "headline", "sentiment", "confidence", "score"])

    tokenizer, model, device = load_sentiment_agent()
    records = []

    if tokenizer is not None and model is not None and torch is not None:
        texts = [h for _, h in headlines]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for (dt_str, text), p in zip(headlines, probs):
            neg_p = float(p[0])
            pos_p = float(p[-1]) if len(p) > 1 else 0.0
            score = float(np.clip(pos_p - neg_p, -1.0, 1.0))
            conf = float(np.max(p))
            if score > 0.05:
                label = "positive"
            elif score < -0.05:
                label = "negative"
            else:
                label = "neutral"
            records.append(
                {
                    "date": dt_str,
                    "headline": text,
                    "sentiment": label,
                    "confidence": round(conf, 4),
                    "score": round(score, 4),
                }
            )
    else:
        pos_words = {"beat", "beats", "surge", "surges", "gain", "gains", "record", "upgrade", "buy", "rally", "strong"}
        neg_words = {"miss", "misses", "drop", "drops", "fall", "falls", "downgrade", "cut", "cuts", "loss", "slide", "risk"}
        for dt_str, text in headlines:
            words = set(text.lower().split())
            p_hits = len(words & pos_words)
            n_hits = len(words & neg_words)
            score = float(np.clip((p_hits - n_hits) * 0.45, -1.0, 1.0))
            label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
            conf = min(0.99, 0.55 + 0.15 * abs(p_hits - n_hits))
            records.append(
                {
                    "date": dt_str,
                    "headline": text,
                    "sentiment": label,
                    "confidence": round(conf, 4),
                    "score": round(score, 4),
                }
            )

    df_out = pd.DataFrame(records)
    avg_score = float(df_out["score"].mean()) if not df_out.empty else 0.0
    return float(np.clip(avg_score, -1.0, 1.0)), df_out


def predict_agent3(prob_up: float, sentiment_score: float) -> float:
    combiner = load_combiner_model()
    if combiner is not None:
        try:
            x = np.array([[prob_up, sentiment_score]], dtype=np.float64)
            return float(np.clip(combiner.predict_proba(x)[0, 1], 0.0, 1.0))
        except Exception:
            pass
    z = (0.6175 * prob_up) + (0.1340 * sentiment_score) - 0.3022
    return float(1.0 / (1.0 + np.exp(-z)))


# ---------------------------------------------------------------------------
# Market & Live News Data Fetchers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_history(ticker: str, as_of: date, lookback_days: int) -> pd.DataFrame:
    sym = ticker.strip().upper()
    if not sym:
        return pd.DataFrame()
    start_dt = (as_of - timedelta(days=lookback_days + 25)).isoformat()
    end_dt = (as_of + timedelta(days=1)).isoformat()
    try:
        df = yf.download(sym, start=start_dt, end=end_dt, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        return compute_technical_features(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_headlines(ticker: str, limit: int = 20) -> List[Tuple[str, str]]:
    sym = ticker.strip().upper()
    if not sym:
        return []
    rows: List[Tuple[str, str]] = []
    try:
        tk = yf.Ticker(sym)
        news_list = getattr(tk, "news", None) or []
        for entry in news_list[:limit]:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content", entry) if isinstance(entry.get("content"), dict) else entry
            title = str(content.get("title") or entry.get("title") or "").strip()
            if not title:
                continue
            pub_raw = content.get("pubDate") or entry.get("providerPublishTime") or ""
            dt_str = ""
            if isinstance(pub_raw, (int, float)) and pub_raw > 0:
                dt_str = pd.to_datetime(pub_raw, unit="s", utc=True).strftime("%Y-%m-%d")
            elif pub_raw:
                parsed = pd.to_datetime(pub_raw, errors="coerce", utc=True)
                if pd.notna(parsed):
                    dt_str = parsed.strftime("%Y-%m-%d")
            rows.append((dt_str, title))
    except Exception:
        pass
    return rows


@st.cache_data(ttl=300, show_spinner=False)
def fetch_google_news_headlines(ticker: str, limit: int = 20) -> List[Tuple[str, str]]:
    sym = ticker.strip().upper()
    if not sym:
        return []
    query = urllib.parse.quote(f'"{sym}" stock')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    rows: List[Tuple[str, str]] = []
    for item in root.findall(".//channel/item")[:limit]:
        title_el = item.find("title")
        pub_el = item.find("pubDate")
        if title_el is None or not title_el.text:
            continue
        raw_title = title_el.text.strip()
        headline = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title
        if not headline:
            continue
        dt_str = ""
        if pub_el is not None and pub_el.text:
            parsed = pd.to_datetime(pub_el.text.strip(), errors="coerce", utc=True)
            if pd.notna(parsed):
                dt_str = parsed.strftime("%Y-%m-%d")
        rows.append((dt_str, headline))
    return rows


def get_live_headlines(
    ticker: str,
    source_mode: str = "Both (Yahoo + Google News)",
    limit: int = 20,
) -> List[Tuple[str, str]]:
    combined: List[Tuple[str, str]] = []
    if source_mode in ("Yahoo Finance", "Both (Yahoo + Google News)"):
        combined.extend(fetch_yahoo_headlines(ticker, limit=limit))
    if source_mode in ("Google News RSS", "Both (Yahoo + Google News)"):
        combined.extend(fetch_google_news_headlines(ticker, limit=limit))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for dt_str, hl in combined:
        norm = " ".join(hl.lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append((dt_str, hl))
    return deduped[:limit]


def parse_uploaded_headlines(uploaded_file) -> List[Tuple[str, str]]:
    if uploaded_file is None:
        return []
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return []
    if df.empty:
        return []

    headline_col = None
    for cand in ["headline", "title", "Headline", "Title", "text"]:
        if cand in df.columns:
            headline_col = cand
            break
    if headline_col is None:
        headline_col = df.columns[0]

    date_col = None
    for cand in ["date", "Date", "pubDate", "published_at", "timestamp"]:
        if cand in df.columns:
            date_col = cand
            break

    rows: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        hl = str(row.get(headline_col) or "").strip()
        if not hl:
            continue
        dt_str = ""
        if date_col is not None:
            parsed = pd.to_datetime(row.get(date_col), errors="coerce", utc=True)
            if pd.notna(parsed):
                dt_str = parsed.strftime("%Y-%m-%d")
        rows.append((dt_str, hl))
    return rows


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Market Data")
    market_source = st.selectbox(
        "Market Source",
        options=["Live Fetch (yfinance)", "Upload Market CSV"],
        index=0,
    )
    ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper()
    as_of_date = st.date_input("As-Of Date", value=date(2026, 11, 27))
    lookback_days = st.selectbox("Lookback Window (Days)", options=[30, 60, 90, 180], index=2)
    uploaded_market_csv = st.file_uploader("Upload Market CSV", type=["csv"])

    st.header("2. News Headlines")
    fetch_live_news = st.checkbox("Fetch live headlines", value=True)
    news_source = st.selectbox(
        "Live News Source",
        options=["Both (Yahoo + Google News)", "Google News RSS", "Yahoo Finance"],
        index=0,
        disabled=not fetch_live_news,
    )
    uploaded_headlines_csv = st.file_uploader("Upload Headlines CSV", type=["csv"])
    custom_headlines_text = st.text_area(
        "Custom Headlines (one per line)",
        placeholder="Optional headlines for current run...",
        height=100,
    )

    st.header("3. Agent 3 Decision")
    threshold = st.slider("UP Decision Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.01)
    run_clicked = st.button("Run Agent Decisions", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------
st.title("Quantitative Trading Swarm — Live & Uploaded Data Decisions")

market_df = pd.DataFrame()
if market_source == "Upload Market CSV" and uploaded_market_csv is not None:
    try:
        raw_mkt = pd.read_csv(uploaded_market_csv)
        market_df = compute_technical_features(raw_mkt)
    except Exception as exc:
        st.error(f"Failed to parse uploaded market CSV: {exc}")
else:
    market_df = fetch_market_history(ticker, as_of_date, lookback_days)

if not market_df.empty:
    cutoff = pd.Timestamp(as_of_date, tz="UTC")
    market_df = market_df[market_df["Date"] <= cutoff].tail(lookback_days).reset_index(drop=True)

active_headlines: List[Tuple[str, str]] = []
if fetch_live_news and ticker:
    active_headlines.extend(get_live_headlines(ticker, source_mode=news_source))

active_headlines.extend(parse_uploaded_headlines(uploaded_headlines_csv))

if custom_headlines_text.strip():
    as_of_str = as_of_date.isoformat()
    for line in custom_headlines_text.splitlines():
        clean = line.strip()
        if clean:
            active_headlines.append((as_of_str, clean))

if market_df.empty:
    st.warning(
        "No market bars available for the selected ticker/date. "
        "Switch **Market Source** to **Upload Market CSV** or verify network access to Yahoo Finance."
    )
    st.stop()

prob_up = predict_agent1(market_df)
sentiment_score, headlines_df = score_agent2(active_headlines)
combined_prob = predict_agent3(prob_up, sentiment_score)

a1_label = "UP" if prob_up >= 0.50 else "DOWN"
if sentiment_score > 0.05:
    a2_label = "POSITIVE"
elif sentiment_score < -0.05:
    a2_label = "NEGATIVE"
else:
    a2_label = "NEUTRAL"
a3_label = "UP" if combined_prob >= threshold else "DOWN"

# Top 3 Agent Cards
c1, c2, c3 = st.columns(3)

with c1:
    badge_cls = "" if a1_label == "UP" else "card-badge-down"
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="card-label">Agent 1 (Market)</div>
            <div class="card-value">{a1_label}</div>
            <span class="card-badge {badge_cls}">&#8593; Prob Up: {prob_up:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    if a2_label == "POSITIVE":
        badge_cls = ""
    elif a2_label == "NEGATIVE":
        badge_cls = "card-badge-down"
    else:
        badge_cls = "card-badge-neutral"
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="card-label">Agent 2 (News)</div>
            <div class="card-value">{a2_label}</div>
            <span class="card-badge {badge_cls}">&#8593; Score: {sentiment_score:+.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    badge_cls = "" if a3_label == "UP" else "card-badge-down"
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="card-label">Agent 3 (Final Decision)</div>
            <div class="card-value">{a3_label}</div>
            <span class="card-badge {badge_cls}">&#8593; Combined Up: {combined_prob:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Price + MA + RSI Chart
chart_title = f"Uploaded CSV ({ticker}) Price & Moving Averages" if market_source == "Upload Market CSV" else f"Live Fetch ({ticker}) Price & Moving Averages"
fig_price = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.06,
    subplot_titles=(chart_title, "RSI (14)"),
)

fig_price.add_trace(
    go.Candlestick(
        x=market_df["Date"],
        open=market_df["Open"],
        high=market_df["High"],
        low=market_df["Low"],
        close=market_df["Close"],
        name="OHLC",
    ),
    row=1,
    col=1,
)
fig_price.add_trace(
    go.Scatter(x=market_df["Date"], y=market_df["MA_5"], mode="lines", name="MA 5"),
    row=1,
    col=1,
)
fig_price.add_trace(
    go.Scatter(x=market_df["Date"], y=market_df["MA_20"], mode="lines", name="MA 20"),
    row=1,
    col=1,
)
fig_price.add_trace(
    go.Scatter(x=market_df["Date"], y=market_df["RSI_14"], mode="lines", name="RSI 14"),
    row=2,
    col=1,
)
fig_price.add_hline(y=70, line_dash="dash", line_color="#64748b", row=2, col=1)
fig_price.add_hline(y=30, line_dash="dash", line_color="#64748b", row=2, col=1)
fig_price.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(fig_price, use_container_width=True)

# Signal Comparison & Headline Polarity Charts
left_col, right_col = st.columns(2)

scaled_sent = float(np.clip((sentiment_score + 1.0) / 2.0, 0.0, 1.0))
with left_col:
    fig_bars = go.Figure(
        data=[
            go.Bar(
                x=["Agent 1 (Prob Up)", "Agent 2 (Scaled Sent.)", "Agent 3 (Combined)"],
                y=[prob_up, scaled_sent, combined_prob],
                text=[f"{prob_up:.3f}", f"{scaled_sent:.3f}", f"{combined_prob:.3f}"],
                textposition="outside",
            )
        ]
    )
    fig_bars.add_hline(y=threshold, line_dash="dash", line_color="#94a3b8")
    fig_bars.update_layout(
        title="Agent Ratings on [0.0 - 1.0] Scale",
        yaxis=dict(range=[0.0, 1.05]),
        height=340,
        template="plotly_dark",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

with right_col:
    if not headlines_df.empty:
        idx_labels = [str(i + 1) for i in range(len(headlines_df))]
        fig_pol = go.Figure(
            data=[
                go.Bar(
                    x=idx_labels,
                    y=headlines_df["score"],
                    text=headlines_df["sentiment"],
                    textposition="outside",
                )
            ]
        )
        fig_pol.update_layout(
            title="Headline Polarity Scores (-1 to +1)",
            yaxis=dict(range=[-1.15, 1.15]),
            xaxis_title="Headline #",
            height=340,
            template="plotly_dark",
        )
        st.plotly_chart(fig_pol, use_container_width=True)
    else:
        st.info("No active headlines to plot polarity scores.")

# Active Headlines Table
st.subheader("Active Headlines & Agent 2 Ratings")
if not headlines_df.empty:
    st.dataframe(headlines_df, use_container_width=True, hide_index=True)
else:
    st.info("No active headlines loaded for this run.")