import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from utils import TICKERS, SEQ_LEN, RISK_COLS, load_and_split_stock


def main():
    os.makedirs("models/agent3", exist_ok=True)
    scalers = {}
    X_tr_list, y_tr_list, X_va_list, y_va_list, X_te_list, y_te_list = [], [], [], [], [], []

    for ticker in TICKERS:
        train_df, val_df, test_df = load_and_split_stock(ticker)

        scaler = StandardScaler()
        tr_s = scaler.fit_transform(train_df[RISK_COLS].iloc[SEQ_LEN:].values)
        va_s = scaler.transform(val_df[RISK_COLS].iloc[SEQ_LEN:].values)
        te_s = scaler.transform(test_df[RISK_COLS].iloc[SEQ_LEN:].values)
        scalers[ticker] = scaler

        X_tr_list.append(tr_s); y_tr_list.append(train_df["Is_Safe"].iloc[SEQ_LEN:].values)
        X_va_list.append(va_s); y_va_list.append(val_df["Is_Safe"].iloc[SEQ_LEN:].values)
        X_te_list.append(te_s); y_te_list.append(test_df["Is_Safe"].iloc[SEQ_LEN:].values)

    X_train, y_train = np.vstack(X_tr_list), np.concatenate(y_tr_list)
    X_val, y_val = np.vstack(X_va_list), np.concatenate(y_va_list)
    X_test, y_test = np.vstack(X_te_list), np.concatenate(y_te_list)

    best_c, best_val_auc = 1.0, 0.0
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc, best_c = val_auc, c

    risk_model = LogisticRegression(C=best_c, class_weight="balanced", random_state=42)
    risk_model.fit(X_train, y_train)

    test_probs = risk_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.55).astype(int)

    print(f"Best C: {best_c} | Test Accuracy: {accuracy_score(y_test, test_preds):.2%} | Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
    print(classification_report(y_test, test_preds, target_names=["Crash Risk (0)", "Safe (1)"]))

    for feat, weight in zip(RISK_COLS, risk_model.coef_[0]):
        print(f"Learned Weight -> {feat}: {weight:+.4f}")

    joblib.dump({"model": risk_model, "scalers": scalers}, "models/agent3/risk_logreg.pkl")
    print("Saved models/risk_logreg.pkl")


if __name__ == "__main__":
    main()