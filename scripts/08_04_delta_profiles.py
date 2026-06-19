#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_utils import pair_from_id


def main():
    p = argparse.ArgumentParser(description="Compute top-minus-bottom activity profiles per species and feature.")
    p.add_argument("--profiles", required=True, help="TSV: species,sequence_id,feature_id,position columns")
    p.add_argument("--labels", required=True); p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--output", required=True); args = p.parse_args()
    profiles = pd.read_csv(args.profiles, sep="\t"); profiles["pair"] = profiles.sequence_id.map(pair_from_id)
    labels = pd.read_csv(args.labels, sep="\t"); labels["pair"] = labels.geneA.astype(str) + "-" + labels.geneB.astype(str)
    merged = profiles.merge(labels[["species", "pair", "cor_log"]], on=["species", "pair"])
    pos = [c for c in profiles if c.isdigit()]; rows = []
    for species, sub in merged.groupby("species"):
        order = sub[["pair", "cor_log"]].drop_duplicates().sort_values("cor_log"); k = int(len(order) * args.fraction)
        if k < 1: continue
        low, high = set(order.head(k).pair), set(order.tail(k).pair)
        for feature, fsub in sub.groupby("feature_id"):
            # Missing FIMO hits are zeros, so divide sums by the full group size.
            high_mean = fsub[fsub.pair.isin(high)][pos].sum() / k
            low_mean = fsub[fsub.pair.isin(low)][pos].sum() / k
            delta = high_mean - low_mean
            rows.append([species, feature, *delta.fillna(0).to_numpy(dtype=float)])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["species", "feature_id"] + pos).to_csv(out, sep="\t", index=False)


if __name__ == "__main__": main()
