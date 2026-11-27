import os
import pandas as pd

NEWS_CSV = "data/analyst_ratings_processed.csv"
STOCKS_DIR = "data/stocks"
OUT_MATCHED_CSV = "data/matched_headlines.csv"


def load_valid_stock_dates(stocks_dir: str) -> pd.DataFrame:
    pairs = []
    for fname in sorted(os.listdir(stocks_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        ticker = os.path.splitext(fname)[0].upper()
        sdf = pd.read_csv(os.path.join(stocks_dir, fname), usecols=["Date"])
        sdf["date"] = pd.to_datetime(sdf["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        sdf = sdf.dropna(subset=["date"]).drop_duplicates(subset=["date"])
        sdf["ticker"] = ticker
        pairs.append(sdf[["ticker", "date"]])
    return pd.concat(pairs, ignore_index=True)


def main():
    valid_stock_dates = load_valid_stock_dates(STOCKS_DIR)

    df = None
    for enc in ["utf-8", "latin-1"]:
        try:
            df = pd.read_csv(NEWS_CSV, encoding=enc, low_memory=False)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        raise ValueError(f"Could not read {NEWS_CSV}")

    col_map = {c.lower().strip(): c for c in df.columns}
    h_col = next(col_map[k] for k in ["title", "headline", "news", "clean_news"] if k in col_map)
    t_col = next(col_map[k] for k in ["stock", "ticker", "symbol"] if k in col_map)
    d_col = next(col_map[k] for k in ["date", "timestamp", "time", "published_at"] if k in col_map)

    news = pd.DataFrame({
        "ticker": df[t_col].astype(str).str.upper().str.strip(),
        "date": pd.to_datetime(df[d_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize(),
        "headline": df[h_col].astype(str).str.strip(),
    }).dropna()
    news = news[news["headline"] != ""]

    matched = (
        pd.merge(news, valid_stock_dates, on=["ticker", "date"], how="inner")
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUT_MATCHED_CSV) or ".", exist_ok=True)
    matched.to_csv(OUT_MATCHED_CSV, index=False)

    print(f"Saved {len(matched):,} matched headlines -> {OUT_MATCHED_CSV}")
    print(f"Unique (ticker, date) pairs: {len(matched[['ticker', 'date']].drop_duplicates()):,}")
    if not matched.empty:
        print(f"Date range: {matched['date'].min().date()} -> {matched['date'].max().date()}")
        print("\nMatched rows per ticker:")
        print(matched["ticker"].value_counts())
        print("\nSample matched headlines:")
        print(matched.head(10).to_string(index=False))


if __name__ == "__main__":
    main()