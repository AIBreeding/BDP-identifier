#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from scipy.stats import spearmanr as scipy_spearmanr
except ImportError:
    scipy_spearmanr = None


def main():
    p = argparse.ArgumentParser(description="Correlate SAE delta profiles with TFBS or nucleosome delta profiles.")
    p.add_argument("--sae", required=True); p.add_argument("--regulatory", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--output", required=True); args = p.parse_args()
    sae = pd.read_csv(args.sae, sep="\t"); regulatory = pd.read_csv(args.regulatory, sep="\t")
    positions = sorted({c for c in sae if c.isdigit()} & {c for c in regulatory if c.isdigit()}, key=int)
    if len(positions) < 3: raise ValueError("At least three shared position columns are required")
    rows = []
    for species in sorted(set(sae.species) & set(regulatory.species)):
        for _, sae_row in sae[sae.species == species].iterrows():
            sae_vector = sae_row[positions].to_numpy(dtype=float)
            if np.ptp(sae_vector) == 0: continue
            for _, reg_row in regulatory[regulatory.species == species].iterrows():
                reg_vector = reg_row[positions].to_numpy(dtype=float)
                if np.ptp(reg_vector) == 0: continue
                if scipy_spearmanr is not None:
                    rho, pvalue = scipy_spearmanr(sae_vector, reg_vector, nan_policy="omit")
                else:
                    rho = np.corrcoef(pd.Series(sae_vector).rank(), pd.Series(reg_vector).rank())[0, 1]
                    pvalue = np.nan
                if np.isfinite(rho) and abs(rho) >= args.threshold:
                    rows.append([species, sae_row.feature_id, reg_row.feature_id, rho, pvalue, len(positions)])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["species", "sae_feature", "regulatory_feature", "rho", "pvalue", "n_positions"]
    pd.DataFrame(rows, columns=columns).to_csv(out, sep="\t", index=False)


if __name__ == "__main__": main()
