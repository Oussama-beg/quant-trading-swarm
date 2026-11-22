# Quantitative Trading Swarm & Risk Management

Multi-agent quantitative trading and risk management system (**Project 2 — Data Mining**) combining deep learning time-series forecasting, financial transformer sentiment analysis, and supervised volatility risk filtering across 20 US equities.

## Architecture & Agents

| Agent | Model / Method | Input Features | Role in Swarm |
| :--- | :--- | :--- | :--- |
| **Agent 1 (Market)** | Pooled LSTM (`models/agent1/lstm_market_agent.keras`) | 20-day sequences of `Return`, `Close_MA5_Ratio`, `Close_MA20_Ratio`, `MA5_MA20_Ratio`, `Daily_Spread`, `RSI_14`, `MACD_Norm`, `MACD_Hist_Norm` | Predicts next-day upward price direction $P(\text{Up})$ |
| **Agent 2 (Sentiment)** | Fine-tuned DistilBERT (`models/agent2/`) | Financial news headlines | Scores headline sentiment (`positive`, `neutral`, `negative`) $\in [-1, +1]$ |
| **Agent 3 (Risk)** | Balanced Logistic Regression (`models/agent3/risk_logreg.pkl`) | `Volatility_7d`, `Volatility_21d`, `Daily_Spread`, `Return_Lag1` | Predicts crash-safety probability $P(\text{Return}_{t+1} > -1.5\%)$ and acts as a risk circuit breaker |
| **Swarm Orchestrator** | Weighted Consensus & Risk Gate (`src/backtest_swarm.py`, `src/app.py`) | Agent probabilities, sentiment scores & customizable influence weights | Combines agent signals into `BUY`, `HOLD`, or `SELL` decisions |

---

## Repository Structure

```text
QUANTITATIVE-TRADING-SWARM/
├── data/
│   ├── stocks/                      # Historical OHLCV CSVs for 20 tickers (Jackson Crow dataset)
│   └── cleaned_news.csv             # Financial news headlines dataset (Kaggle)
├── models/
│   ├── agent1/
│   │   ├── lstm_market_agent.keras
│   │   └── scaler_market.pkl
│   ├── agent2/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── agent3/
│       └── risk_logreg.pkl
├── results/
│   └── backtest_summary.csv         # Out-of-sample test metrics across all 20 tickers
├── src/
│   ├── utils.py                     # Unified data loader, feature engineering, 70/15/15 split & sequence builder
│   ├── Student1_Market_Agent.ipynb  # Agent 1 LSTM training & evaluation notebook
│   ├── news_agent.py                # Agent 2 DistilBERT inference script
│   ├── agent3_risk.py               # Agent 3 Logistic Regression training & evaluation
│   ├── backtest_swarm.py            # Out-of-sample historical swarm backtester (Agents 1 & 3)
│   └── app.py                       # Interactive Streamlit forecasting & analytics dashboard
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare datasets in `data/`:**
   * Place historical stock CSVs inside `data/stocks/<TICKER>.csv` (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `NFLX`, `INTC`, `CSCO`, `ORCL`, `IBM`, `JPM`, `BAC`, `V`, `GS`, `JNJ`, `PFE`, `WMT`, `KO`, `DIS`, `ADBE`).
   * Place the news headlines dataset at `data/cleaned_news.csv`.

---

## Usage

### 1. Train Individual Agents
* **Agent 1 (Market LSTM):** Run `src/Student1_Market_Agent.ipynb`
* **Agent 3 (Risk Filter):**
  ```bash
  PYTHONPATH=src python src/agent3_risk.py
  ```

### 2. Run Historical Out-of-Sample Backtest (Test Split)
Evaluates **Swarm (Market + Risk Gate)** vs. **Agent 1 Only** vs. **Buy & Hold** across all 20 tickers (with 5 bps transaction costs):
```bash
PYTHONPATH=src python src/backtest_swarm.py --risk-th 0.45
```

### 3. Launch Interactive Streamlit Dashboard
Supports live Yahoo Finance OHLCV & headline pulling, custom OHLC CSV uploads, hypothetical next-bar simulation, customizable agent influence weights, and Plotly visual analytics:
```bash
python -m streamlit run src/app.py
```

---

## Key Empirical Results (Test Split — 20 Tickers)

* **Standalone Market Direction (`Target_Up`):** Single-agent next-day direction accuracy is `0.483` (`AUC = 0.502`), reflecting high daily return noise and regime shifts (including the 2020 market crash).
* **Swarm Drawdown Protection (`--risk-th 0.45`):** Integrating Agent 3's volatility risk gate cuts average Maximum Drawdown across the 20-stock test split from **-39.52%** (Buy & Hold) and **-9.79%** (Market LSTM Only) down to **-7.10%**, preserving capital during severe downturns (e.g., `IBM` and `GS`).