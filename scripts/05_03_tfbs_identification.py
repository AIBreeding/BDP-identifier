#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
from Bio import motifs
from Bio.Seq import Seq
from feature_utils import read_fasta


def main():
    p = argparse.ArgumentParser(); p.add_argument("--fasta", required=True); p.add_argument("--motifs", required=True)
    p.add_argument("--format", choices=["meme", "jaspar"], default="meme")
    p.add_argument("--threshold", type=float, default=5.0); p.add_argument("--pseudocount", type=float, default=1e-4)
    p.add_argument("--output", required=True); args = p.parse_args()
    with open(args.motifs) as fh: motif_list = list(motifs.parse(fh, args.format))
    pssms = [(m.name or m.matrix_id, m.counts.normalize(pseudocounts=args.pseudocount).log_odds()) for m in motif_list]
    rows = []
    for sequence_id, sequence in read_fasta(args.fasta):
        row = {"gene_id": sequence_id}; seq = Seq(sequence)
        for name, pssm in pssms:
            scores = np.r_[pssm.calculate(seq), pssm.reverse_complement().calculate(seq)]
            count = int(np.sum(scores >= args.threshold)); row[f"{name}_count"] = count
            row[f"{name}_density"] = count / len(sequence)
        rows.append(row)
    pd.DataFrame(rows).to_pickle(args.output)


if __name__ == "__main__": main()
