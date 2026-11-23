
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# 1. Load the cleaned dataset
df = pd.read_csv("data/cleaned_news.csv")

df = df.dropna(subset=["clean_news", "sentiment"])
df["clean_news"] = df["clean_news"].astype(str).str.strip()
df = df[df["clean_news"] != ""]

# 2. Prepare texts and labels
label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

id2label = {v: k for k, v in label2id.items()}

X = df["clean_news"].tolist()
y = df["sentiment"].str.lower().map(label2id)

# Remove any unexpected labels
valid = y.notna()
X = [text for text, keep in zip(X, valid) if keep]
y = y[valid].astype(int).tolist()

# 3. Same train/test split as ComplementNB
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Load DistilBERT tokenizer
MODEL_NAME = "distilbert-base-uncased"

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 5. Tokenize the news
train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=128
)

# 6. Create PyTorch dataset
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = NewsDataset(train_encodings, y_train)
test_dataset = NewsDataset(test_encodings, y_test)

# 7. Load pretrained DistilBERT
print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# 8. Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions)
    }


# 9. Training configuration
training_args = TrainingArguments(
    output_dir="models/distilbert_checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    use_cpu=True
)

# 10. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("\nStarting DistilBERT training...")
trainer.train()

# 11. Evaluate
print("\nEvaluating model...")
predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["negative", "neutral", "positive"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

# 12. Save the trained model
SAVE_PATH = "models/distilbert_news_sentiment"

trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\nModel saved successfully to:", SAVE_PATH)
