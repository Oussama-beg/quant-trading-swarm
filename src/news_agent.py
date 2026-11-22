

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "models/agent2"

print("Loading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")



def analyze_headlines(headlines, batch_size=16):

    results = []

    sentiment_values = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    for start in range(0, len(headlines), batch_size):

        batch = headlines[start:start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        predictions = torch.argmax(probabilities, dim=1)

        for i, prediction in enumerate(predictions):

            sentiment = model.config.id2label[
                prediction.item()
            ].lower()

            confidence = probabilities[i][prediction].item()

            results.append({
                "predicted_sentiment": sentiment,
                "confidence": round(confidence, 4),
                "score": sentiment_values[sentiment]
            })

        print(
            f"Analyzed {min(start + batch_size, len(headlines))}"
            f" / {len(headlines)} headlines"
        )

    return results



def analyze_single_headline(headline):

    result = analyze_headlines([headline], batch_size=1)[0]

    print("\n========== SINGLE HEADLINE ANALYSIS ==========")
    print("Headline:", headline)
    print("Predicted sentiment:", result["predicted_sentiment"].upper())
    print("Confidence:", f'{result["confidence"] * 100:.2f}%')
    print("Sentiment score:", result["score"])

    return result


def analyze_dataset():

    print("\nLoading financial news dataset...")

    df = pd.read_csv("data/cleaned_news.csv", encoding="latin-1")

    # Handle headerless 2-column Kaggle file (sentiment, headline) vs named columns
    if "clean_news" in df.columns:
        text_col = "clean_news"
    elif "news" in df.columns:
        text_col = "news"
    elif df.shape[1] == 2:
        df = pd.read_csv("data/cleaned_news.csv", encoding="latin-1", header=None, names=["sentiment", "clean_news"])
        text_col = "clean_news"
    else:
        # Fallback: pick the longest string column as the headline text
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())

    df = df.dropna(subset=[text_col]).copy()
    df["clean_news"] = df[text_col].astype(str).str.strip()
    df = df[df["clean_news"] != ""].reset_index(drop=True)

    headlines = df["clean_news"].tolist()

    print("Total headlines:", len(headlines))

    predictions = analyze_headlines(headlines)
    predictions_df = pd.DataFrame(predictions)

    df["predicted_sentiment"] = predictions_df["predicted_sentiment"]
    df["confidence"] = predictions_df["confidence"]
    df["score"] = predictions_df["score"]

    sentiment_score = df["score"].mean()

    if sentiment_score > 0.2:
        signal = "BUY"
    elif sentiment_score < -0.2:
        signal = "SELL"
    else:
        signal = "HOLD"

    sentiment_counts = df["predicted_sentiment"].value_counts()

    output_path = "data/analyzed_news.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n========== NEWS ANALYSIS SUMMARY ==========")
    print("Total headlines:", len(df))
    print("\nSentiment distribution:")
    print(sentiment_counts)
    print("\nAverage sentiment score:", round(sentiment_score, 4))
    print("Overall signal:", signal)
    print("\nResults saved to:", output_path)
  

if __name__ == "__main__":

    analyze_dataset()

    test_headlines = [
        "The company reported record profits and strong revenue growth.",
        "The company announced its annual general meeting.",
        "The company suffered significant financial losses."
    ]

    for headline in test_headlines:
        analyze_single_headline(headline)

