import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    csv_path = "data/aligned_agent_outputs.csv"
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    df = df.dropna(subset=["prob_up", "sentiment_score", "target_up"]).reset_index(drop=True)
    y = (df["target_up"].to_numpy(dtype=float) > 0).astype(int)

    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)

    train_df = df.iloc[:i1].copy()
    val_df = df.iloc[i1:i2].copy()
    test_df = df.iloc[i2:].copy()

    y_train = y[:i1]
    y_val = y[i1:i2]
    y_test = y[i2:]

    X_train = train_df[["prob_up", "sentiment_score"]].to_numpy(dtype=float)
    X_val = val_df[["prob_up", "sentiment_score"]].to_numpy(dtype=float)
    X_test = test_df[["prob_up", "sentiment_score"]].to_numpy(dtype=float)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)

    print("============================================================")
    print("AGENT 3 EVALUATION METRICS REPORT (2-FEATURE BASELINE)")
    print("============================================================")
    print(f"Total aligned rows : {n}")
    print(f"Split sizes        : train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Target Up rate     : full={y.mean():.4f}, train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}")
    print(f"Majority Up share  : full={y.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}")
    print(f"Learned weights    : w_prob_up={clf.coef_[0][0]:.6f}, w_sentiment={clf.coef_[0][1]:.6f}, intercept={clf.intercept_[0]:.6f}")

    for split_name, X_s, y_s in [
        ("VALIDATION", X_val, y_val),
        ("TEST", X_test, y_test),
    ]:
        probs = clf.predict_proba(X_s)[:, 1]
        preds = (probs >= 0.50).astype(int)
        majority_pred = int(round(float(y_s.mean())))
        majority_acc = float((np.full_like(y_s, majority_pred) == y_s).mean())
        auc = roc_auc_score(y_s, probs) if len(np.unique(y_s)) > 1 else float("nan")
        acc = accuracy_score(y_s, preds)
        bal_acc = balanced_accuracy_score(y_s, preds)
        macro_f1 = f1_score(y_s, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_s, preds, average="weighted", zero_division=0)
        down_f1 = f1_score(y_s, preds, pos_label=0, zero_division=0)
        up_f1 = f1_score(y_s, preds, pos_label=1, zero_division=0)
        down_p = precision_score(y_s, preds, pos_label=0, zero_division=0)
        up_p = precision_score(y_s, preds, pos_label=1, zero_division=0)
        down_r = recall_score(y_s, preds, pos_label=0, zero_division=0)
        up_r = recall_score(y_s, preds, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_s, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"\n--- {split_name} SET METRICS (threshold = 0.50, n = {len(y_s)}) ---")
        print(f"Up class share     : {y_s.mean():.4f} (majority baseline acc = {majority_acc:.4f})")
        print(f"ROC-AUC           : {auc:.4f}")
        print(f"Accuracy          : {acc:.4f}")
        print(f"Balanced Accuracy : {bal_acc:.4f}")
        print(f"Macro F1          : {macro_f1:.4f}")
        print(f"Weighted F1       : {weighted_f1:.4f}")
        print(f"Down [0]          : precision={down_p:.4f}, recall={down_r:.4f}, f1={down_f1:.4f}, support={int((y_s == 0).sum())}")
        print(f"Up [1]            : precision={up_p:.4f}, recall={up_r:.4f}, f1={up_f1:.4f}, support={int((y_s == 1).sum())}")
        print(f"Confusion Matrix  : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_s, preds, target_names=["Down [0]", "Up [1]"], digits=4, zero_division=0))

        comb_margin = np.abs(probs - 0.50)
        prob_margin = np.abs(X_s[:, 0] - 0.50)
        sent_mag = np.abs(X_s[:, 1])

        print(f"--- {split_name} SUBSETS BY SIGNAL MAGNITUDE ---")
        for label, mask in [
            ("|prob_up - 0.5| > 0.03", prob_margin > 0.03),
            ("|prob_up - 0.5| > 0.05", prob_margin > 0.05),
            ("|sentiment| > 0.10", sent_mag > 0.10),
            ("|sentiment| > 0.25", sent_mag > 0.25),
            ("|sentiment| > 0.50", sent_mag > 0.50),
            ("|comb_prob - 0.5| > 0.02", comb_margin > 0.02),
        ]:
            cnt = int(mask.sum())
            if cnt > 0:
                sub_acc = float((preds[mask] == y_s[mask]).mean())
                sub_up_rate = float(preds[mask].mean())
                cov = cnt / len(y_s)
                print(f"{label:26s} : n={cnt:4d} (cov={cov:.4f}), acc={sub_acc:.4f}, pred_up_rate={sub_up_rate:.4f}")
            else:
                print(f"{label:26s} : n=   0 (cov=0.0000), acc=nan, pred_up_rate=nan")


if __name__ == "__main__":
    main()