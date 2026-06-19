#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_utils import pair_from_id, read_fasta
from model_utils import extreme_labels, score


def encode(sequences):
    mapping = {b: i for i, b in enumerate("ACGT")}; x = np.zeros((len(sequences), 4, len(sequences[0])), dtype=np.float32)
    for n, seq in enumerate(sequences):
        for i, base in enumerate(seq):
            if base in mapping: x[n, mapping[base], i] = 1
    return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(4, 64, 8), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 6), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--labels", required=True); p.add_argument("--fasta", required=True)
    p.add_argument("--label-ratio", type=float, default=0.10); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--output", required=True); args = p.parse_args()
    torch.manual_seed(42); np.random.seed(42); device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = pd.DataFrame(read_fasta(args.fasta), columns=["id", "sequence"]); seq["pair"] = seq.id.map(pair_from_id)
    data = extreme_labels(args.labels, args.label_ratio).merge(seq, on="pair"); X = encode(data.sequence.tolist()); y = data.label.to_numpy()
    rows = []
    for fold, (train, test) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(X, y), 1):
        model = CNN().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(X[train]), torch.from_numpy(y[train]).float()), batch_size=args.batch_size, shuffle=True)
        for _ in range(args.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad(); loss = loss_fn(model(xb.to(device)), yb.to(device)); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): prob = torch.sigmoid(model(torch.from_numpy(X[test]).to(device))).cpu().numpy()
        rows.append({"fold": fold, **score(y[test], prob)})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__": main()
