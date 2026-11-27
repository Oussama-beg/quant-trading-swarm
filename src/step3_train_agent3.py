import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.combiner_agent import DEFAULT_MODEL_PATH, save_combiner

ALIGNED_CSV = "data/aligned_agent_outputs.csv"
METRICS_OUT = "models/agent3/metrics.json"
CANDIDATE_CS = [1.0, 0.1, 0.01]
CANDIDATE_WEIGHTS = [None, "balanced"]


def eval_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "auc": auc,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "down_recall": float(recall_score(y_true, preds, pos_label=0, zero_division=0)),
        "up_recall": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "pred_up_rate": float(preds.mean()),
        "true_up_rate": float(y_true.mean()),
    }


def main():
    aligned = pd.read_csv(ALIGNED_CSV, parse_dates=["date"])
    aligned = aligned.sort_values(["date", "ticker"]).reset_index(drop=True)
    if len(aligned) < 10:
        raise RuntimeError(f"Only {len(aligned)} aligned rows found; need at least 10.")

    split = int(len(aligned) * 0.80)
    train_df, test_df = aligned.iloc[:split], aligned.iloc[split:]

    X_train = train_df[["prob_up", "sentiment_score"]].to_numpy()
    y_train = train_df["target_up"].to_numpy()
    X_test = test_df[["prob_up", "sentiment_score"]].to_numpy()
    y_test = test_df["target_up"].to_numpy()

    a1_metrics = eval_predictions(y_test, test_df["prob_up"].to_numpy(), threshold=0.5)
    print(
        f"Agent 1 Alone -> Acc: {a1_metrics['accuracy']:.4f} | AUC: {a1_metrics['auc']:.4f} | "
        f"Macro F1: {a1_metrics['macro_f1']:.4f} | DOWN Rec: {a1_metrics['down_recall']:.4f} | "
        f"Pred UP%: {a1_metrics['pred_up_rate']:.1%}"
    )

    best_score = -1.0
    best_bundle = None

    for cw in CANDIDATE_WEIGHTS:
        for c_val in CANDIDATE_CS:
            clf = LogisticRegression(C=c_val, class_weight=cw, max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            m = eval_predictions(y_test, probs, threshold=0.5)
            m.update({
                "C": c_val,
                "class_weight": cw if cw is not None else "none",
                "coef_agent1": float(clf.coef_[0][0]),
                "coef_agent2": float(clf.coef_[0][1]),
                "intercept": float(clf.intercept_[0]),
                "agent1_accuracy": a1_metrics["accuracy"],
                "agent1_auc": a1_metrics["auc"],
            })
            print(
                f"C={c_val:<4} cw={str(cw):<8} -> "
                f"Acc: {m['accuracy']:.4f} | AUC: {m['auc']:.4f} | Macro F1: {m['macro_f1']:.4f} | "
                f"DOWN Rec: {m['down_recall']:.4f} | UP Rec: {m['up_recall']:.4f} | "
                f"Pred UP%: {m['pred_up_rate']:.1%} | "
                f"w1={m['coef_agent1']:+.4f}, w2={m['coef_agent2']:+.4f}, b={m['intercept']:+.4f}"
            )
            rank_key = np.nan_to_num(m["auc"], nan=-1.0) + m["macro_f1"]
            if rank_key > best_score:
                best_score = rank_key
                best_bundle = (clf, m)

    best_clf, best_metrics = best_bundle
    saved_path = save_combiner(best_clf, DEFAULT_MODEL_PATH)
    os.makedirs(os.path.dirname(METRICS_OUT) or ".", exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(
        f"\nSaved best model (C={best_metrics['C']}, class_weight={best_metrics['class_weight']}) "
        f"-> {saved_path} | metrics -> {METRICS_OUT}"
    )


if __name__ == "__main__":
    main()