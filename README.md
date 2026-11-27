# Quantitative Trading Swarm & Risk Management

Multi-agent quantitative trading decision-support system (**Project 2 — Data Mining**) combining deep learning time-series forecasting, financial transformer sentiment analysis, and calibrated logistic regression signal fusion across 20 US equities and live market/news feeds.

## Architecture & Agents

| Agent | Model / Method | Input Features | Role in Swarm |
| :--- | :--- | :--- | :--- |
| **Agent 1 (Market)** | Pooled LSTM (`models/agent1/lstm_market_agent.keras`) | 20-day sequences of `Return`, `Close_MA5_Ratio`, `Close_MA20_Ratio`, `MA5_MA20_Ratio`, `Daily_Spread`, `RSI_14`, `MACD_Norm`, `MACD_Hist_Norm` | Predicts next-day upward price probability $P(\text{Up})$ (`UP` / `DOWN`) |
| **Agent 2 (News Sentiment)** | Fine-tuned DistilBERT (`models/agent2/`) | Financial news headlines (WordPiece max length 64) | Classifies headlines (`positive`, `neutral`, `negative`) and maps them to a continuous polarity score $\in [-1.0, +1.0]$ |
| **Agent 3 (Signal Combiner)** | Balanced Logistic Regression (`C = 1.0`, `class_weight="balanced"`) | Date-aligned Agent 1 `prob_up` and Agent 2 daily average `sentiment_score` | Fuses technical and sentiment signals into a calibrated probability $P(\text{Up})$ (`UP` when $\ge \text{threshold}$, default `0.50`) |
| **Interactive Dashboard** | Streamlit Decision Engine (`src/app.py`) | Live `yfinance` OHLCV, deduplicated Yahoo Finance & Google News RSS headlines, or custom CSV uploads | Runs all 3 agents in real time with an adjustable **UP Decision Threshold** slider (`0.10`–`0.90`) and interactive Plotly charts |

---

## Datasets

1. **[Jackson Crow Stock Market Dataset (Kaggle)](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset):** Historical daily OHLCV records across 20 US equities (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `NFLX`, `INTC`, `CSCO`, `ORCL`, `IBM`, `JPM`, `BAC`, `V`, `GS`, `JNJ`, `PFE`, `WMT`, `KO`, `DIS`, `ADBE`) used to compute the 8 technical indicators and next-day directional targets.
2. **[Sentiment Analysis for Financial News (Kaggle)](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news):** Labeled financial news headlines used to fine-tune the 3-class DistilBERT sentiment classifier.
3. **[Daily Financial News for 6000+ Stocks (Kaggle)](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests):** Date-stamped stock headlines indexed by ticker and publication date, aligned on `(ticker, date)` with Agent 1 outputs to construct the `8,682`-row chronological dataset (`70/15/15` split: `6,077` train / `1,302` validation / `1,303` test) for training and evaluating Agent 3.

---

## Repository Structure

```text
QUANTITATIVE-TRADING-SWARM/
├── data/
│   ├── stocks/                      # Historical OHLCV CSVs for 20 tickers (Jackson Crow dataset)
│   ├── cleaned_news.csv             # Labeled financial news headlines dataset
│   └── dated_stock_news.csv         # Date-stamped ticker headlines for Agent 3 alignment
├── models/
│   ├── agent1/
│   │   ├── lstm_market_agent.keras
│   │   └── scaler_market.pkl
│   ├── agent2/
│   │   ├── config.json
│   │   ├── model.safetensors          # Hosted externally on Google Drive (see Setup)
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── agent3/                      # Saved Balanced Logistic Regression combiner artifacts
├── src/
│   ├── utils.py                     # Unified data loader, 8 technical features, 70/15/15 split & sequence builder
│   ├── Student1_Market_Agent.ipynb  # Agent 1 Pooled LSTM training & evaluation notebook
│   ├── news_agent.py                # Agent 2 DistilBERT inference script
│   ├── eval_metric_logreg.py        # Agent 3 Balanced Logistic Regression training, validation & test evaluation
│   └── app.py                       # Interactive Streamlit dashboard (live yfinance, Yahoo/Google News RSS, CSV uploads)
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare datasets & model weights:**
   - Place historical stock CSVs inside `data/stocks/<TICKER>.csv` (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `NFLX`, `INTC`, `CSCO`, `ORCL`, `IBM`, `JPM`, `BAC`, `V`, `GS`, `JNJ`, `PFE`, `WMT`, `KO`, `DIS`, `ADBE`).
   - Place the labeled sentiment headlines CSV and date-stamped stock news CSV inside `data/`.
   - Download `model.safetensors` (255 MB) from Google Drive (`https://drive.google.com/file/d/16TCelOauewhTi5pqD2z8cYA-Uqa31Ozg/view?usp=sharing`) and place it at `models/agent2/model.safetensors`.

---

## Usage

### 1. Train & Evaluate Individual Agents
- **Agent 1 (Pooled LSTM — Training & Evaluation):** Run `src/Student1_Market_Agent.ipynb`
- **Agent 2 (Fine-Tuned DistilBERT — Standalone Inference):**
  ```bash
  PYTHONPATH=src python src/news_agent.py
  ```
- **Agent 3 (Balanced Logistic Regression Combiner — Training & Evaluation):**
  ```bash
  PYTHONPATH=src python src/eval_metric_logreg.py
  ```

### 2. Launch Interactive Streamlit Dashboard
Runs all 3 agents together with API-key-free live market data (`yfinance`), deduplicated live news feeds (**Both (Yahoo + Google News)**, **Google News RSS**, or **Yahoo Finance**), custom market/headline CSV uploads, an adjustable **UP Decision Threshold** slider (`0.10`–`0.90`), and interactive Plotly charts (OHLC + `MA 5` / `MA 20`, `RSI (14)`, normalized `[0.0–1.0]` agent ratings, and per-headline polarity scores):
```bash
python -m streamlit run src/app.py
```

---

## Key Empirical Results

- **Agent 1 Standalone (Pooled LSTM — 27,432 Test Sequences):** Achieves `0.483` accuracy (vs. `0.522` majority baseline) and `0.502` ROC-AUC, exhibiting class-prior collapse toward `DOWN` predictions (`97.2%` of calls) due to daily return noise and regime shifts.
- **Agent 3 Signal Combiner (Balanced Logistic Regression — 8,682 Aligned Rows):**
  - **Learned Weights:** `w_prob_up = 0.720855`, `w_sentiment = 0.107888`, `intercept = -0.349348`.
  - **Validation Set (`n = 1,302`, threshold `0.50`):** `0.5399` accuracy (above the `0.5131` majority baseline), `0.5355` ROC-AUC, and `0.5393` Macro F1.
  - **Test Set (`n = 1,303`, threshold `0.50`):** `0.4873` accuracy (`0.4943` ROC-AUC, `0.4872` Macro F1) with balanced confusion matrix counts (`TN = 309`, `FP = 333`, `FN = 335`, `TP = 326`), eliminating Agent 1's single-class collapse.
  - **High-Conviction Test Subset (`|comb_prob - 0.5| > 0.02`, `n = 158`, `12.13%` coverage):** Accuracy rises to `0.5506` (`pred_up_rate = 0.4367`), demonstrating a clear precision–coverage trade-off on higher-conviction signals.