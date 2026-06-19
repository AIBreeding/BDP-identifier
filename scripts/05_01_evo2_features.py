#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evo2 import Evo2
from feature_utils import read_fasta


def main():
    p = argparse.ArgumentParser(description="Mean-pool Evo2 hidden states for DNA sequences.")
    p.add_argument("--fasta", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="evo2_7b")
    p.add_argument("--layers", nargs="+", type=int, default=[16, 18, 20, 22, 24, 26, 28, 30])
    args = p.parse_args()
    records = read_fasta(args.fasta); model = Evo2(args.model)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    for layer in args.layers:
        layer_name = f"blocks.{layer}.mlp.l3"; rows = []
        for sequence_id, seq in records:
            tokens = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0)
            if torch.cuda.is_available(): tokens = tokens.cuda()
            with torch.no_grad(): _, hidden = model(tokens, return_embeddings=True, layer_names=[layer_name])
            vector = hidden[layer_name].float().mean(dim=1).squeeze(0).cpu().numpy()
            rows.append(vector)
        df = pd.DataFrame(np.stack(rows)); df.insert(0, "gene_id", [x[0] for x in records])
        with (outdir / f"{Path(args.fasta).stem}_layer{layer}.pkl").open("wb") as fh: pickle.dump(df, fh)


if __name__ == "__main__": main()
