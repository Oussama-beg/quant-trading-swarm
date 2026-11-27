# Quantitative Trading Swarm & Risk Management

Multi-agent quantitative trading decision-support system (**Project 2 — Data Mining**) combining deep learning time-series forecasting, financial transformer sentiment analysis, and calibrated logistic regression signal fusion across 20 US equities and live market/news feeds.

## Architecture & Agents

| Agent | Model / Method | Input Features | Role in Swarm |
| :--- | :--- | :--- | :--- |
| **Agent 1 (Market)** | Pooled LSTM (`models/agent1/lstm_market_agent.keras`) | 20-day sequences of `Return`, `Close_MA5_Ratio`, `Close_MA20_Ratio`, `MA5_MA20_Ratio`, `Daily_Spread`, `RSI_14`, `MACD_Norm`, `MACD_Hist_Norm` | Predicts next-day upward price probability $P(\text{Up})$ (`UP` / `DOWN`) |
| **Agent 2 (News Sentiment)** | Fine-tuned DistilBERT (`models/agent2/`) | Financial news headlines (WordPiece max length 64) | Classifies headlines (`positive`, `neutral`, `negative`) and maps them to a continuous polarity score $\in [-1.0, +1.0]$ |
| **Agent 3 (Signal Combiner)** | Balanced Logistic Regression (`models/agent3/combiner_lr.pkl`, `C = 1.0`, `class_weight="balanced"`) | Date-aligned Agent 1 `prob_up` and Agent 2 daily average `sentiment_score` | Fuses technical and sentiment signals into a calibrated probability $P(\text{Up})$ (`UP` when $\ge \text{threshold}$, default `0.50`) |
| **Interactive Dashboard** | Streamlit Decision Engine (`src/app.py`) | Live `yfinance` OHLCV, deduplicated Yahoo Finance & Google News RSS headlines, or custom CSV uploads | Runs all 3 agents in real time with an adjustable **UP Decision Threshold** slider (`0.10`–`0.90`) and interactive Plotly charts |

---

## Datasets & Subsets Used

1. **[Jackson Crow Stock Market Dataset (Kaggle)](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset):**
   - **Subset used:** Only the `stocks/` folder for **20 high-liquidity US equities** (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `NFLX`, `INTC`, `CSCO`, `ORCL`, `IBM`, `JPM`, `BAC`, `V`, `GS`, `JNJ`, `PFE`, `WMT`, `KO`, `DIS`, `ADBE`), excluding ETFs and remaining tickers.
   - **Columns & split:** Daily `Date`, `Open`, `High`, `Low`, `Close`, and `Volume` across each ticker's full history, converted into 8 normalized technical indicators and 20-day rolling windows (`129,333` train / `27,413` validation / `27,432` test sequences using a per-ticker chronological `70/15/15` split).
2. **[Sentiment Analysis for Financial News (Kaggle)](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news):**
   - **Subset used:** The `all-data.csv` headline corpus (`Sentiment` label + `News Headline` text), cleaned and stored in `data/cleaned_news.csv`.
   - **Columns & split:** Used strictly for fine-tuning Agent 2 (`DistilBERT`) on 3-class sentiment (`positive`, `neutral`, `negative`) with a stratified `70/15/15` split (since this corpus does not include trading dates).
3. **[Daily Financial News for 6000+ Stocks (Kaggle)](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests):**
   - **Subset used:** Specifically the **`analyst_ratings_processed.csv`** file from the multi-file Kaggle archive (ignoring the raw/unprocessed splits), filtered to the **20 target tickers** (`stock` column) and overlapping trading dates (`date` column) in `src/step0_match_news.py`. *(Note: `analyst_ratings_processed.csv` is ~151 MB and excluded from Git, while the resulting `matched_headlines.csv` and aligned outputs are included in `data/`.)*
   - **Columns & split:** Scored with Agent 2 (`src/step1_score_news.py` → `data/daily_news_scores.csv`), joined with Agent 1 `prob_up` outputs (`src/step2_score_stocks.py` → `data/daily_stock_scores.csv` & `data/aligned_agent_outputs.csv`) to form **`8,682` aligned `(ticker, date)` rows**, partitioned chronologically `70/15/15` (`6,077` train / `1,302` validation / `1,303` test) for Agent 3 (`src/step3_train_agent3.py`, `src/eval_metric_logreg.py`).

---

## Repository Structure

```text
QUANTITATIVE-TRADING-SWARM/
├── data/
│   ├── stocks/                       # Historical OHLCV CSVs for the 20 target tickers
│   ├── analyst_ratings_processed.csv # Raw Kaggle dated headlines file (151 MB; download from Kaggle to re-run step 0)
│   ├── cleaned_news.csv              # Cleaned labeled financial news headlines for Agent 2
│   ├── matched_headlines.csv         # Filtered headlines matching the 20 tickers & trading dates
│   ├── daily_news_scores.csv         # Daily average Agent 2 sentiment scores per (ticker, date)
│   ├── daily_stock_scores.csv        # Daily Agent 1 upward probabilities per (ticker, date)
│   └── aligned_agent_outputs.csv     # Final 8,682 date-aligned rows used to train/evaluate Agent 3
├── models/
│   ├── agent1/
│   │   ├── lstm_market_agent.keras
│   │   └── scaler_market.pkl
│   ├── agent2/
│   │   ├── config.json
│   │   ├── model.safetensors           # Hosted externally on Google Drive (see Setup)
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── agent3/
│       ├── combiner_lr.pkl
│       └── metrics.json
├── src/
│   ├── utils.py                      # Unified data loader, 8 technical features, 70/15/15 split & sequence builder
│   ├── Student1_Market_Agent.ipynb   # Agent 1 Pooled LSTM training & evaluation notebook
│   ├── news_agent.py                 # Agent 2 DistilBERT inference script
│   ├── step0_match_news.py           # Filters analyst_ratings_processed.csv to the 20 target tickers
│   ├── step1_score_news.py           # Scores matched headlines with Agent 2 into daily sentiment
│   ├── step2_score_stocks.py         # Generates Agent 1 daily probabilities & aligns on (ticker, date)
│   ├── step3_train_agent3.py         # Trains Balanced Logistic Regression combiner (combiner_lr.pkl)
│   ├── eval_metric_logreg.py         # Evaluates Agent 3 across validation, test & signal-magnitude subsets
│   └── app.py                        # Interactive Streamlit dashboard (live yfinance, Yahoo/Google News RSS, CSV uploads)
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
   - The 20 ticker OHLCV files (`data/stocks/<TICKER>.csv`) and the processed alignment CSVs (`cleaned_news.csv`, `matched_headlines.csv`, `daily_news_scores.csv`, `daily_stock_scores.csv`, `aligned_agent_outputs.csv`) are already included in `data/`.
   - *(Optional — only required if re-running `src/step0_match_news.py` from scratch)* Download **`analyst_ratings_processed.csv`** from the [Daily Financial News for 6000+ Stocks](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) Kaggle dataset and place it at `data/analyst_ratings_processed.csv`.
   - Download `model.safetensors` (255 MB) from Google Drive (`https://drive.google.com/file/d/16TCelOauewhTi5pqD2z8cYA-Uqa31Ozg/view?usp=sharing`) and place it at `models/agent2/model.safetensors`.

---

## Usage

### 1. Train & Evaluate Individual Agents
- **Agent 1 (Pooled LSTM — Training & Evaluation):** Run `src/Student1_Market_Agent.ipynb`
- **Agent 2 (Fine-Tuned DistilBERT — Standalone Inference):**
  ```bash
  PYTHONPATH=src python src/news_agent.py
  ```
- **Agent 3 (Dataset Alignment, Combiner Training & Evaluation):**
  ```bash
  # Optional if data/aligned_agent_outputs.csv is already present:
  PYTHONPATH=src python src/step0_match_news.py
  PYTHONPATH=src python src/step1_score_news.py
  PYTHONPATH=src python src/step2_score_stocks.py

  # Train and evaluate Agent 3:
  PYTHONPATH=src python src/step3_train_agent3.py
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