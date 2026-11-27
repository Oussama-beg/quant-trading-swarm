import os
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import TECH_COLS, SEQ_LEN, find_data_dir, load_and_split_stock, make_sequences

NEWS_SCORES_CSV = "data/daily_news_scores.csv"
AGENT1_DIR = "models/agent1"
OUT_CSV = "data/daily_stock_scores.csv"


def main():
    if not os.path.exists(NEWS_SCORES_CSV):
        raise FileNotFoundError(f"Run step1_score_news.py first to generate {NEWS_SCORES_CSV}")

    daily_news = pd.read_csv(NEWS_SCORES_CSV, parse_dates=["date"])
    target_tickers = sorted(daily_news["ticker"].unique())
    min_date = daily_news["date"].min()
    max_date = daily_news["date"].max()

    lstm_model = tf.keras.models.load_model(os.path.join(AGENT1_DIR, "lstm_market_agent.keras"))
    scaler = joblib.load(os.path.join(AGENT1_DIR, "scaler_market.pkl"))
    data_dir = find_data_dir()

    records = []
    for ticker in target_tickers:
        t_up = ticker.upper()
        try:
            tr, va, te = load_and_split_stock(t_up, data_dir)
            feat_df = pd.concat([tr, va, te], ignore_index=True)
        except Exception:
            continue

        if feat_df.empty or not {"Date", "Target_Up"}.issubset(feat_df.columns):
            continue

        feat_df["Date"] = pd.to_datetime(feat_df["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        feat_df = (
            feat_df.dropna(subset=["Date", "Target_Up"])
            .sort_values("Date")
            .drop_duplicates(subset=["Date"])
            .reset_index(drop=True)
        )
        if len(feat_df) <= SEQ_LEN:
            continue

        scaled_df = feat_df.copy()
        scaled_df[TECH_COLS] = np.clip(scaler.transform(scaled_df[TECH_COLS]), -5, 5)

        X_all, y_all = make_sequences(scaled_df, TECH_COLS, "Target_Up")
        n_seq = len(X_all)
        if n_seq == 0:
            continue

        seq_dates = scaled_df["Date"].iloc[-n_seq:].reset_index(drop=True)
        mask = (
            (seq_dates >= min_date - pd.Timedelta(days=5))
            & (seq_dates <= max_date + pd.Timedelta(days=5))
        ).to_numpy()
        if not mask.any():
            continue

        X_sub = X_all[mask]
        y_sub = np.asarray(y_all)[mask]
        d_sub = seq_dates.iloc[mask].reset_index(drop=True)
        probs = lstm_model.predict(X_sub, batch_size=512, verbose=0).ravel()

        for dt, p, yt in zip(d_sub, probs, y_sub):
            records.append({
                "ticker": t_up,
                "date": pd.Timestamp(dt),
                "prob_up": float(p),
                "target_up": int(yt),
            })

    stock_scores = pd.DataFrame(records, columns=["ticker", "date", "prob_up", "target_up"])
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    stock_scores.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(stock_scores):,} daily stock rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()