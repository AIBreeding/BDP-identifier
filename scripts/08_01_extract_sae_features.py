#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from evo2 import Evo2

from feature_utils import read_fasta


class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(self, d_in, expansion_factor, k):
        super().__init__(); self.k = k
        self.W = torch.nn.Parameter(torch.empty(d_in, d_in * expansion_factor))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_in * expansion_factor))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
    def encode(self, x):
        values = torch.relu(x @ self.W + self.b_enc)
        top = torch.topk(values, min(self.k, values.shape[-1]), dim=-1)
        return torch.zeros_like(values).scatter(-1, top.indices, top.values)


def main():
    p = argparse.ArgumentParser(description="Extract position-resolved Evo2 SAE features.")
    p.add_argument("--fasta", required=True); p.add_argument("--sae-checkpoint", required=True)
    p.add_argument("--model", default="evo2_7b"); p.add_argument("--layer", type=int, default=24)
    p.add_argument("--d-in", type=int, default=4096); p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--top-k", type=int, default=64); p.add_argument("--output", required=True); args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; model = Evo2(args.model)
    sae = BatchTopKTiedSAE(args.d_in, args.expansion_factor, args.top_k)
    raw = torch.load(args.sae_checkpoint, weights_only=True, map_location="cpu")
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw.items()}
    sae.load_state_dict(state); sae.to(device).eval(); layer_name = f"blocks.{args.layer}.mlp.l3"
    result = {}
    for sequence_id, seq in read_fasta(args.fasta):
        tokens = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)
        with torch.no_grad():
            _, hidden = model(tokens, return_embeddings=True, layer_names=[layer_name])
            result[sequence_id] = sae.encode(hidden[layer_name][0].float()).cpu().numpy().astype(np.float32)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh: pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__": main()
