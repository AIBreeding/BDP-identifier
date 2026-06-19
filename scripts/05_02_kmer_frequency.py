#!/usr/bin/env python3
import argparse
from itertools import product

import pandas as pd
from feature_utils import read_fasta


def main():
    p = argparse.ArgumentParser(); p.add_argument("--fasta", required=True); p.add_argument("--output", required=True)
    p.add_argument("--k", nargs="+", type=int, default=[2, 3, 4]); args = p.parse_args()
    records = read_fasta(args.fasta); rows = []
    columns = ["".join(x) for k in args.k for x in product("ACGT", repeat=k)]
    for sequence_id, seq in records:
        row = {x: 0.0 for x in columns}
        for k in args.k:
            valid = [seq[i:i+k] for i in range(len(seq)-k+1) if set(seq[i:i+k]) <= set("ACGT")]
            denom = len(valid)
            if denom:
                for word in valid: row[word] += 1.0 / denom
        row["gene_id"] = sequence_id; rows.append(row)
    pd.DataFrame(rows, columns=["gene_id"] + columns).to_pickle(args.output)


if __name__ == "__main__": main()
