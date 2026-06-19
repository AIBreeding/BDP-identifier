#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from model_utils import extreme_labels, load_features, score


def main():
    p = argparse.ArgumentParser(description="Evaluate Evo2, k-mer, or TFBS features with LightGBM.")
    p.add_argument("--labels", required=True); p.add_argument("--feature", action="append", required=True,
        help="NAME=feature.pkl; repeat for multiple feature sets")
    p.add_argument("--label-ratios", nargs="+", type=float, default=[i / 100 for i in range(1, 15)])
    p.add_argument("--learning-rates", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.07, 0.09])
    p.add_argument("--folds", type=int, default=5); p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", required=True); args = p.parse_args()
    results = []
    for spec in args.feature:
        name, path = spec.split("=", 1); feat = load_features(path)
        for ratio in args.label_ratios:
            labels = extreme_labels(args.labels, ratio); data = labels.merge(feat, on="pair")
            X = data.drop(columns=["species", "pair", "label", "cor_log"]); y = data.label.to_numpy()
            for lr in args.learning_rates:
                fold_scores = []; cv = StratifiedKFold(args.folds, shuffle=True, random_state=42)
                for train, test in cv.split(X, y):
                    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=lr, num_leaves=127,
                        max_depth=9, min_child_samples=30, reg_alpha=1.0, reg_lambda=1.0,
                        colsample_bytree=0.7, subsample=0.8, subsample_freq=3,
                        random_state=42, n_jobs=args.threads, verbosity=-1)
                    model.fit(X.iloc[train], y[train]); fold_scores.append(score(y[test], model.predict_proba(X.iloc[test])[:, 1]))
                row = {"feature": name, "label_ratio": ratio, "learning_rate": lr, "n": len(y)}
                for metric in ("AUROC", "F1", "accuracy"):
                    values = [x[metric] for x in fold_scores]; row[f"{metric}_mean"] = np.mean(values); row[f"{metric}_sd"] = np.std(values)
                results.append(row)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, sep="\t", index=False)


if __name__ == "__main__": main()
