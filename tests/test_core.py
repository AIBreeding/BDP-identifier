import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from feature_utils import pair_from_id, read_fasta
from model_utils import extreme_labels


def test_fasta_and_pair(tmp_path):
    path = tmp_path / "x.fa"
    path.write_text(">Rice|A-B|chr1:1-4\nACGT\n")
    assert read_fasta(path) == [("Rice|A-B|chr1:1-4", "ACGT")]
    assert pair_from_id("Rice|A-B|chr1:1-4") == "A-B"


def test_extreme_labels_are_balanced(tmp_path):
    rows = [{"species": "Rice", "geneA": f"A{i}", "geneB": f"B{i}", "cor_log": i} for i in range(100)]
    path = tmp_path / "labels.tsv"; pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    labels = extreme_labels(path, 0.1)
    assert labels.label.value_counts().to_dict() == {0: 10, 1: 10}
