"""Training script for Spam-Classifier.

Lightweight CLI to train a logistic regression with TF-IDF + heuristic features.
Saves a pipeline (vectorizers + model) to disk for inference.
"""
import argparse
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from src.preprocess import extract_features


def train(
    data_path: str,
    text_col: str = "text",
    label_col: str = "label",
    out_dir: str = "models",
    sample: int = 0,
):
    df = pd.read_csv(data_path)
    if sample and sample > 0:
        df = df.sample(n=sample, random_state=42)

    X, wv, cv = extract_features(df, text_col=text_col, fit=True)
    y = df[label_col].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)

    model = LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        model.fit(Xtr, ytr)
        oof[te] = model.predict_proba(Xte)[:, 1]
        print(f"Fold {fold} done")

    auc = roc_auc_score(y, oof)
    print("OOF ROC AUC:", auc)
    preds = (oof >= 0.5).astype(int)
    print(classification_report(y, preds))

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "word_vec": wv, "char_vec": cv}, Path(out_dir) / "model.joblib")
    print(f"Saved model bundle to {out_dir}/model.joblib")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", help="CSV file with text and label columns")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--out_dir", default="models")
    p.add_argument("--sample", type=int, default=0, help="If >0, train on a sample of rows for quick runs")
    args = p.parse_args()
    train(args.data, text_col=args.text_col, label_col=args.label_col, out_dir=args.out_dir, sample=args.sample)


if __name__ == "__main__":
    main()
