"""Préparation des données — Projet 2 (commune à toute l'équipe).
Base = code de l'équipe (split chronologique 70/15/15 par action).
Ajouts Student 1 : RSI et MACD (demandés dans le sujet)."""
import os
import numpy as np
import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "NFLX", "INTC", "CSCO", "ORCL", "IBM",
    "JPM", "BAC", "V", "GS",
    "JNJ", "PFE", "WMT", "KO", "DIS",
    "ADBE",
]

SEQ_LEN = 20

# Student 1 : ajout de RSI_14, MACD_Norm, MACD_Hist_Norm
TECH_COLS = ["Return", "Close_MA5_Ratio", "Close_MA20_Ratio", "MA5_MA20_Ratio", "Daily_Spread",
             "RSI_14", "MACD_Norm", "MACD_Hist_Norm"]
RISK_COLS = ["Volatility_7d", "Volatility_21d", "Daily_Spread", "Return_Lag1"]


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def load_and_split_stock(ticker, data_dir="data", train_ratio=0.70, val_ratio=0.15):
    df = pd.read_csv(f"{data_dir}/stocks/{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Guard against any zero/corrupt prices in very old rows
    df = df[(df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0)].reset_index(drop=True)

    # Core price & moving average indicators
    df["Return"] = df["Close"].pct_change()
    df["Return_Lag1"] = df["Return"].shift(1)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    # Stationary technical ratios (scale-invariant across decades)
    df["Close_MA5_Ratio"] = df["Close"] / df["MA_5"] - 1.0
    df["Close_MA20_Ratio"] = df["Close"] / df["MA_20"] - 1.0
    df["MA5_MA20_Ratio"] = df["MA_5"] / df["MA_20"] - 1.0

    # --- Ajout Student 1 : RSI & MACD (normalisés, sans échelle) ---
    df["RSI_14"] = _rsi(df["Close"]) / 100.0
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Norm"] = macd / df["Close"]
    df["MACD_Hist_Norm"] = (macd - macd_sig) / df["Close"]

    # Multi-horizon volatility & intraday range features for Risk Agent
    df["Volatility_7d"] = df["Return"].rolling(7).std()
    df["Volatility_21d"] = df["Return"].rolling(21).std()
    df["Daily_Spread"] = (df["High"] - df["Low"]) / df["Close"]

    # Targets
    df["Next_Return"] = df["Return"].shift(-1)
    df["Target_Up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Is_Safe"] = (df["Next_Return"] > -0.015).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df["Ticker"] = ticker

    # Chronological 70% Train / 15% Validation / 15% Test split per ticker
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def find_data_dir():
    """Cherche automatiquement le dossier qui contient 'stocks/' (local, Colab, Kaggle)."""
    for d in ["data", "/content/data"]:
        if os.path.isdir(f"{d}/stocks"):
            return d
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "stocks" in dirs:
            return root
    raise FileNotFoundError("Dossier 'stocks/' introuvable. Mets-le dans data/stocks/ ou ajoute le dataset (Add Input).")


def make_sequences(df, cols, target, seq_len=SEQ_LEN):
    """Fenêtres glissantes DANS une seule action (jamais à cheval sur 2 actions)."""
    X = df[cols].values.astype("float32")
    y = df[target].values
    idx = range(seq_len - 1, len(df))
    Xs = np.stack([X[i - seq_len + 1:i + 1] for i in idx])
    ys = np.array([y[i] for i in idx])
    return Xs, ys
