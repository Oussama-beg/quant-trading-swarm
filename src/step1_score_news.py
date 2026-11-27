import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MATCHED_CSV = "data/matched_headlines.csv"
AGENT2_DIR = "models/agent2"
OUT_CSV = "data/daily_news_scores.csv"
BATCH_SIZE = 64


def main():
    if not os.path.exists(MATCHED_CSV):
        raise FileNotFoundError(f"Run step0_match_news.py first to create {MATCHED_CSV}")

    matched_news = pd.read_csv(MATCHED_CSV, parse_dates=["date"])
    matched_news = matched_news.dropna(subset=["ticker", "date", "headline"]).reset_index(drop=True)
    matched_news = matched_news[matched_news["headline"].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"Loaded {len(matched_news):,} matched headlines from {MATCHED_CSV}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(AGENT2_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(AGENT2_DIR).to(device)
    model.eval()

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    neg_idx = next((i for i, l in id2label.items() if "neg" in l), 0)
    pos_idx = next((i for i, l in id2label.items() if "pos" in l), model.config.num_labels - 1)

    scores = []
    texts = matched_news["headline"].astype(str).tolist()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()
        scores.extend((probs[:, pos_idx] - probs[:, neg_idx]).tolist())

    matched_news["sentiment_score"] = scores
    daily_news = matched_news.groupby(["ticker", "date"], as_index=False)["sentiment_score"].mean()

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    daily_news.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(daily_news):,} daily news rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()