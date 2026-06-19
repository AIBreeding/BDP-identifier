#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from feature_utils import pair_from_id


def main():
    p = argparse.ArgumentParser(description="Compute position-resolved SAE top-minus-bottom delta profiles.")
    p.add_argument("--sae", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--min-abs-delta", type=float, default=0.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with open(args.sae, "rb") as fh:
        tensors = pickle.load(fh)
    labels = pd.read_csv(args.labels, sep="\t")
    labels["pair"] = labels.geneA.astype(str) + "-" + labels.geneB.astype(str)
    by_pair = {pair_from_id(key): value for key, value in tensors.items()}
    rows, n_positions = [], 0
    for species, sub in labels.dropna(subset=["cor_log"]).groupby("species"):
        sub = sub[sub.pair.isin(by_pair)].sort_values("cor_log")
        k = int(len(sub) * args.fraction)
        if k < 1:
            continue
        low = np.stack([by_pair[pair] for pair in sub.head(k).pair])
        high = np.stack([by_pair[pair] for pair in sub.tail(k).pair])
        delta = high.mean(axis=0) - low.mean(axis=0)
        n_positions = delta.shape[0]
        keep = np.flatnonzero(np.max(np.abs(delta), axis=0) >= args.min_abs_delta)
        for feature in keep:
            rows.append([species, int(feature), *delta[:, feature].astype(float)])
    columns = ["species", "feature_id"] + [str(i) for i in range(1, n_positions + 1)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
