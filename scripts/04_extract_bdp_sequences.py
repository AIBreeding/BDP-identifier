#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta


def main():
    p = argparse.ArgumentParser(description="Extract fixed windows around BDP intergenic midpoints.")
    p.add_argument("--candidates", required=True)
    p.add_argument("--genomes", required=True, help="TSV with species and fasta columns")
    p.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    candidates = pd.read_csv(args.candidates, sep="\t")
    manifest = pd.read_csv(args.genomes, sep="\t")
    genomes = {r.species: Fasta(r.fasta, sequence_always_upper=True) for r in manifest.itertuples()}
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    for length in args.lengths:
        if length <= 0 or length % 2: raise ValueError("All lengths must be positive even integers")
        written = 0
        with (outdir / f"BDP_center_{length}bp.fasta").open("w") as out:
            for r in candidates.itertuples():
                midpoint0 = ((int(r.endA) - 1) + (int(r.startB) - 1)) // 2
                start0 = midpoint0 - length // 2
                end0 = start0 + length
                genome = genomes[r.species]
                if start0 < 0 or str(r.chr) not in genome or end0 > len(genome[str(r.chr)]):
                    continue
                seq = str(genome[str(r.chr)][start0:end0])
                if len(seq) != length: continue
                pair = f"{r.geneA}-{r.geneB}"
                out.write(f">{r.species}|{pair}|{r.chr}:{start0 + 1}-{end0}\n{seq}\n")
                written += 1
        print(f"{length} bp: wrote {written} sequences")


if __name__ == "__main__": main()
