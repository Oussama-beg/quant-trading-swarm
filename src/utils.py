import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
START_DATE = "2015-01-01"
TRAIN_END  = "2018-06-30"
VAL_END    = "2019-04-30"
SEQ_LEN    = 20

TECH_COLS = ["Close", "Volume", "Return", "MA_5", "MA_20"]
RISK_COLS = ["Volatility_7d", "Daily_Spread"]

def load_and_split_stock(ticker, data_dir="data"):
    df = pd.read_csv(f"{data_dir}/stocks/{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] >= START_DATE].reset_index(drop=True)

    df["Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility_7d"] = df["Return"].rolling(7).std()
    df["Daily_Spread"] = (df["High"] - df["Low"]) / df["Close"]

    df["Next_Return"] = df["Return"].shift(-1)
    df["Target_Up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Is_Safe"] = (df["Next_Return"] > -0.015).astype(int)

    df = df.dropna().reset_index(drop=True)
    df["Ticker"] = ticker

    train_df = df[df["Date"] <= TRAIN_END].reset_index(drop=True)
    val_df   = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)].reset_index(drop=True)
    test_df  = df[df["Date"] > VAL_END].reset_index(drop=True)
    return train_df, val_df, test_df
