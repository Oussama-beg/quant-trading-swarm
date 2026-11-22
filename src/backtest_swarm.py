import os
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import TICKERS, SEQ_LEN, TECH_COLS, RISK_COLS, find_data_dir, load_and_split_stock, make_sequences


def swarm_decision(
    p_up: float,
    p_safe: float,
    sentiment_score: float = 0.0,
    risk_threshold: float = 0.45,
    tech_buy_th: float = 0.52,
    tech_sell_th: float = 0.48,
    sent_th: float = 0.20,
) -> tuple[int, str]:
    """
    Combine Agent 1 (Market LSTM), Agent 2 (News Sentiment, optional), and Agent 3 (Risk LogReg).
    Returns:
        position: 1 (Long), 0 (Cash/Hold)
        reason: Human-readable decision trace
    """
    # 1. Risk Agent Veto Gate
    if p_safe <= risk_threshold:
        return 0, "BLOCKED_BY_RISK"

    # 2. Directional Votes
    tech_vote = 1 if p_up > tech_buy_th else (-1 if p_up < tech_sell_th else 0)
    news_vote = 1 if sentiment_score > sent_th else (-1 if sentiment_score < -sent_th else 0)

    # 3. Consensus Logic (Long-only spot equity backtest)
    if tech_vote == 1 and news_vote >= 0:
        return 1, "BUY_CONSENSUS"
    if news_vote == 1 and tech_vote >= 0:
        return 1, "BUY_NEWS_DRIVEN"
    if tech_vote == -1 or news_vote == -1:
        return 0, "EXIT_BEARISH"
    return 0, "HOLD_NEUTRAL"


def compute_metrics(returns: np.ndarray, trading_days: int = 252) -> dict:
    """Compute cumulative return, annualized Sharpe ratio, and max drawdown."""
    equity = np.cumprod(1.0 + returns)
    cum_return = equity[-1] - 1.0
    std = np.std(returns)
    sharpe = (np.mean(returns) / std) * np.sqrt(trading_days) if std > 1e-9 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = np.min((equity - peak) / peak)
    return {
        "Cum_Return_%": round(cum_return * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Max_Drawdown_%": round(max_dd * 100, 2),
    }


def run_backtest(tx_cost_bps: float = 5.0, risk_threshold: float = 0.45):
    data_dir = find_data_dir()

    # Load Agent 1 (Market LSTM + Scaler)
    market_model = tf.keras.models.load_model("models/agent1/lstm_market_agent.keras")
    market_scaler = joblib.load("models/agent1/scaler_market.pkl")

    # Load Agent 3 (Risk Logistic Regression + Per-Ticker Scalers)
    risk_bundle = joblib.load("models/agent3/risk_logreg.pkl")
    risk_model = risk_bundle["model"]
    risk_scalers = risk_bundle["scalers"]

    cost_rate = tx_cost_bps / 10000.0
    summary_rows = []
    reason_counts = {}

    for ticker in TICKERS:
        _, _, test_df = load_and_split_stock(ticker, data_dir=data_dir)
        if len(test_df) <= SEQ_LEN:
            continue

        # Prepare Agent 1 sequences (indices SEQ_LEN-1 .. len-1)
        tech_df = test_df.copy()
        tech_df[TECH_COLS] = np.clip(market_scaler.transform(tech_df[TECH_COLS]), -5, 5)
        X_seq, _ = make_sequences(tech_df, TECH_COLS, "Target_Up", seq_len=SEQ_LEN)
        p_up_all = market_model.predict(X_seq, verbose=0).ravel()

        # Align test_df rows to sequence end dates (index SEQ_LEN-1 onwards)
        aligned_df = test_df.iloc[SEQ_LEN - 1 :].reset_index(drop=True)

        # Prepare Agent 3 probabilities on exact aligned rows
        X_risk = risk_scalers[ticker].transform(aligned_df[RISK_COLS].values)
        p_safe_all = risk_model.predict_proba(X_risk)[:, 1]

        next_returns = aligned_df["Next_Return"].values
        n_days = len(aligned_df)

        # Strategy 1: Swarm (Agent 1 Market + Agent 3 Risk Gate)
        swarm_pos = np.zeros(n_days)
        # Strategy 2: Agent 1 Market LSTM Only (No Risk Gate)
        tech_only_pos = (p_up_all > 0.52).astype(float)

        for i in range(n_days):
            pos, reason = swarm_decision(
                p_up=p_up_all[i],
                p_safe=p_safe_all[i],
                sentiment_score=0.0,
                risk_threshold=risk_threshold,
            )
            swarm_pos[i] = pos
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Apply transaction costs on position changes
        swarm_turnover = np.abs(np.diff(swarm_pos, prepend=0.0))
        tech_turnover = np.abs(np.diff(tech_only_pos, prepend=0.0))

        swarm_ret = swarm_pos * next_returns - swarm_turnover * cost_rate
        tech_ret = tech_only_pos * next_returns - tech_turnover * cost_rate
        bh_ret = next_returns

        m_swarm = compute_metrics(swarm_ret)
        m_tech = compute_metrics(tech_ret)
        m_bh = compute_metrics(bh_ret)

        summary_rows.append({
            "Ticker": ticker,
            "Days": n_days,
            "Swarm_Ret_%": m_swarm["Cum_Return_%"],
            "Tech_Ret_%": m_tech["Cum_Return_%"],
            "BH_Ret_%": m_bh["Cum_Return_%"],
            "Swarm_Sharpe": m_swarm["Sharpe"],
            "Tech_Sharpe": m_tech["Sharpe"],
            "BH_Sharpe": m_bh["Sharpe"],
            "Swarm_MaxDD_%": m_swarm["Max_Drawdown_%"],
            "Tech_MaxDD_%": m_tech["Max_Drawdown_%"],
            "BH_MaxDD_%": m_bh["Max_Drawdown_%"],
            "Time_In_Market_%": round(swarm_pos.mean() * 100, 1),
        })

    results_df = pd.DataFrame(summary_rows)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/backtest_summary.csv", index=False)

    print("\n================ SWARM BACKTEST SUMMARY (TEST SPLIT) ================")
    print(results_df.to_string(index=False))

    print("\n================ PORTFOLIO AVERAGES ACROSS 20 TICKERS ================")
    print(f"Swarm Avg Return      : {results_df['Swarm_Ret_%'].mean():.2f}%  | Avg Sharpe: {results_df['Swarm_Sharpe'].mean():.3f} | Avg MaxDD: {results_df['Swarm_MaxDD_%'].mean():.2f}%")
    print(f"Tech-Only Avg Return  : {results_df['Tech_Ret_%'].mean():.2f}%  | Avg Sharpe: {results_df['Tech_Sharpe'].mean():.3f} | Avg MaxDD: {results_df['Tech_MaxDD_%'].mean():.2f}%")
    print(f"Buy & Hold Avg Return : {results_df['BH_Ret_%'].mean():.2f}%  | Avg Sharpe: {results_df['BH_Sharpe'].mean():.3f} | Avg MaxDD: {results_df['BH_MaxDD_%'].mean():.2f}%")
    print("\nDecision Breakdown:", reason_counts)
    print("Saved detailed results to results/backtest_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Quantitative Trading Swarm")
    parser.add_argument("--tx-cost-bps", type=float, default=5.0)
    parser.add_argument("--risk-th", type=float, default=0.45)
    args = parser.parse_args()

    run_backtest(tx_cost_bps=args.tx_cost_bps, risk_threshold=args.risk_th)