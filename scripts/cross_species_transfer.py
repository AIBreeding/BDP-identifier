import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from model_utils import extreme_labels, load_features, score, species_combinations


def run(n_train):
    p = argparse.ArgumentParser(); p.add_argument("--labels", required=True); p.add_argument("--features", required=True)
    p.add_argument("--label-ratio", type=float, default=0.10); p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=1); p.add_argument("--output", required=True); args = p.parse_args()
    data = extreme_labels(args.labels, args.label_ratio).merge(load_features(args.features), on="pair")
    feature_cols = data.columns.difference(["species", "pair", "label", "cor_log"]); rows = []
    for training in species_combinations(data.species.unique(), n_train):
        for held_out in sorted(set(data.species.unique()) - set(training)):
            train = data[data.species.isin(training)]; test = data[data.species == held_out]
            model = lgb.LGBMClassifier(n_estimators=300, learning_rate=args.learning_rate, num_leaves=127,
                max_depth=9, min_child_samples=30, reg_alpha=1, reg_lambda=1, colsample_bytree=0.7,
                subsample=0.8, subsample_freq=3, random_state=42, n_jobs=args.threads, verbosity=-1)
            model.fit(train[feature_cols], train.label); metrics = score(test.label, model.predict_proba(test[feature_cols])[:, 1])
            rows.append({"train_species": ",".join(training), "test_species": held_out, "n_train": len(train), "n_test": len(test), **metrics})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
