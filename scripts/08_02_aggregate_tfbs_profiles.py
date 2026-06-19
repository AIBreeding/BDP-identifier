#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Convert FIMO hits to per-sequence motif position profiles.")
    p.add_argument("--fimo", required=True, help="FIMO TSV with sequence_name, motif_alt_id, start, stop")
    p.add_argument("--length", type=int, required=True); p.add_argument("--output", required=True); args = p.parse_args()
    hits = pd.read_csv(args.fimo, sep="\t", comment="#")
    required = {"sequence_name", "motif_alt_id", "start", "stop"}
    if not required <= set(hits): raise ValueError(f"Missing columns: {required - set(hits)}")
    rows = []
    for (seq, motif), sub in hits.groupby(["sequence_name", "motif_alt_id"]):
        profile = [0] * args.length
        for hit in sub.itertuples():
            for pos in range(max(1, int(hit.start)), min(args.length, int(hit.stop)) + 1): profile[pos - 1] += 1
        species = str(seq).split("|")[0]; rows.append([species, seq, motif, *profile])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["species", "sequence_id", "feature_id"] + [str(i) for i in range(1, args.length + 1)]).to_csv(out, sep="\t", index=False)


if __name__ == "__main__": main()
