import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_utils import pair_from_id


def extreme_labels(table, ratio, species=None):
    df = pd.read_csv(table, sep="\t").dropna(subset=["cor_log"])
    if species is not None: df = df[df.species.isin(species)]
    groups = []
    for sp, sub in df.groupby("species"):
        sub = sub.sort_values("cor_log"); k = int(len(sub) * ratio)
        if k < 5: continue
        low, high = sub.head(k).copy(), sub.tail(k).copy()
        low["label"], high["label"] = 0, 1; groups += [low, high]
    if not groups: raise ValueError("No species has at least five samples per class")
    out = pd.concat(groups); out["pair"] = out.geneA.astype(str) + "-" + out.geneB.astype(str)
    return out[["species", "pair", "label", "cor_log"]]


def load_features(path):
    with open(path, "rb") as fh: df = pickle.load(fh)
    if not isinstance(df, pd.DataFrame): df = pd.DataFrame(df)
    id_col = "gene_id" if "gene_id" in df else "sequence name"
    df = df.copy(); df["pair"] = df[id_col].map(pair_from_id)
    numeric = df.drop(columns=[id_col, "pair"]).select_dtypes(include=[np.number])
    return pd.concat([df[["pair"]], numeric], axis=1)


def score(y, probability):
    pred = probability >= 0.5
    return {"AUROC": roc_auc_score(y, probability), "F1": f1_score(y, pred),
            "accuracy": accuracy_score(y, pred)}


def species_combinations(species, n):
    return list(itertools.combinations(sorted(species), n))
