import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from utils import TICKERS, SEQ_LEN, RISK_COLS, load_and_split_stock


def prepare_risk_splits(data_dir="data"):
    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list = [], []

    for ticker in TICKERS:
        train_df, val_df, test_df = load_and_split_stock(ticker, data_dir=data_dir)

        # Slice from SEQ_LEN onward so every row aligns 1-to-1 with Agent 1's LSTM windows
        X_tr_list.append(train_df[RISK_COLS].iloc[SEQ_LEN:].values)
        y_tr_list.append(train_df["Is_Safe"].iloc[SEQ_LEN:].values)

        X_va_list.append(val_df[RISK_COLS].iloc[SEQ_LEN:].values)
        y_va_list.append(val_df["Is_Safe"].iloc[SEQ_LEN:].values)

        X_te_list.append(test_df[RISK_COLS].iloc[SEQ_LEN:].values)
        y_te_list.append(test_df["Is_Safe"].iloc[SEQ_LEN:].values)

    return (
        np.vstack(X_tr_list), np.concatenate(y_tr_list),
        np.vstack(X_va_list), np.concatenate(y_va_list),
        np.vstack(X_te_list), np.concatenate(y_te_list),
    )


def main():
    os.makedirs("models", exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_risk_splits()

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    # Tune regularization C on Validation set (using StandardScaler inside a Pipeline)
    best_c, best_val_auc = 1.0, 0.0
    for c in [0.01, 0.1, 1.0, 10.0]:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", random_state=42)
        )
        pipe.fit(X_train, y_train)
        val_probs = pipe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        if val_auc > best_val_auc:
            best_val_auc, best_c = val_auc, c

    # Train final model with best C
    risk_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, class_weight="balanced", random_state=42)
    )
    risk_model.fit(X_train, y_train)

    # Evaluate on Unseen Test Set (2019-05-01 -> 2020-04-01)
    test_preds = risk_model.predict(X_test)
    test_probs = risk_model.predict_proba(X_test)[:, 1]

    print(f"\nBest C (tuned on Val): {best_c} (Val ROC-AUC: {best_val_auc:.4f})")
    print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.2%}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, test_probs):.4f}")
    print("\nClassification Report (0 = High Crash Risk <= -1.5%, 1 = Safe):")
    print(classification_report(y_test, test_preds, target_names=["Crash Risk (0)", "Safe (1)"]))

    # Inspect learned feature weights
    coefs = risk_model.named_steps["logisticregression"].coef_[0]
    for feat, weight in zip(RISK_COLS, coefs):
        print(f"Learned Weight -> {feat}: {weight:+.4f}")

    # Save pipeline (includes StandardScaler + LogisticRegression together)
    model_path = "models/risk_logreg.pkl"
    joblib.dump(risk_model, model_path)
    print(f"\nSaved Agent 3 Risk Model to {model_path}")


if __name__ == "__main__":
    main()