import numpy as np
import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "NFLX", "INTC", "CSCO", "ORCL", "IBM",
    "JPM", "BAC", "V", "GS",
    "JNJ", "PFE", "WMT", "KO", "DIS",
    "ADBE",
]

SEQ_LEN = 20

TECH_COLS = ["Return", "Close_MA5_Ratio", "Close_MA20_Ratio", "MA5_MA20_Ratio", "Daily_Spread"]
RISK_COLS = ["Volatility_7d", "Volatility_21d", "Daily_Spread", "Return_Lag1"]


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