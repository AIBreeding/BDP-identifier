# BDP-identifier

Identifying plant bidirectional promoter (BDP) candidates, quantifying their activity, generating sequence features, training prediction models, and relating Evo2 sparse-autoencoder (SAE) features to known regulatory signals.

The workflow follows the accompanying manuscript and supports four species: *Arabidopsis thaliana*, *Oryza sativa* (rice), *Zea mays* (maize), and *Triticum aestivum* (wheat).

## Workflow

1. Identify adjacent divergent gene pairs from GFF3 annotations.
2. Use the Pearson correlation of log10-transformed PPRD expression profiles (`cor_log`) as the BDP activity proxy.
3. Merge activity with genomic coordinates and retain high-confidence intergenic candidates.
4. Extract fixed-length DNA windows around each BDP midpoint.
5. Generate Evo2, k-mer, and TFBS features.
6. Compare LightGBM and sequence-CNN models and evaluate cross-species transfer.
7. Extract Evo2-SAE features and correlate high-minus-low activity profiles with TFBS and nucleosome-affinity profiles.

## Repository layout

```text
config/                  Example input manifests
scripts/                 Numbered analysis scripts
tests/                   Small tests for shared data handling
requirements.txt         Core Python dependencies
requirements-evo2.txt    Optional Evo2 dependency
install_r_packages.R     R dependency installer
```

Large genomes, PPRD expression matrices, model weights, JASPAR motifs, and generated results are intentionally not committed. Put local inputs under `data/`; this directory is ignored by Git.

## Installation

Python 3.10+ and R 4.2+ are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Rscript install_r_packages.R
```

Evo2 feature extraction additionally requires the Evo2 package, compatible GPU/PyTorch installation, and model access:

```bash
pip install -r requirements-evo2.txt
```

## Input contracts

**Expression table:** one gene per row, one sample per column, plus a gene identifier column (default `gene_id`). Values must be non-negative. A pair's effective observations are samples in which both genes have finite, positive expression.

**Candidate table:** the output of step 1, including `geneA`, `geneB`, `chr`, `endA`, `startB`, and `distance`.

**Final label table:** includes `species`, `geneA`, `geneB`, and `cor_log`. Feature FASTA headers use:

```text
>Species|geneA-geneB|chromosome:start-end
```

All tabular outputs are TSV unless their filename ends in `.pkl`.

## 1. Identify candidate BDPs

Run the same auditable implementation for each genome. The initial 20-kb threshold reproduces the broad candidate screen; step 3 applies the final 1-kb criterion.

```bash
Rscript scripts/01_identify_bdp_candidates.R --gff data/gff/Arabidopsis.gff3.gz --species Arabidopsis --output results/01_candidates/arabidopsis.tsv
Rscript scripts/01_identify_bdp_candidates.R --gff data/gff/Rice.gff3.gz        --species Rice        --output results/01_candidates/rice.tsv
Rscript scripts/01_identify_bdp_candidates.R --gff data/gff/Maize.gff3.gz       --species Maize       --output results/01_candidates/maize.tsv
Rscript scripts/01_identify_bdp_candidates.R --gff data/gff/Wheat.gff3.gz       --species Wheat       --output results/01_candidates/wheat.tsv
```

## 2. Calculate BDP activity

These four commands retain only `geneA`, `geneB`, `eff_obs`, and `cor_log`. Rice identifiers are uppercased to match the source analysis.

```bash
Rscript scripts/02_calculate_bdp_activity.R --expression data/pprd/arabidopsis.tsv --pairs results/01_candidates/arabidopsis.tsv --output results/02_activity/arabidopsis.tsv
Rscript scripts/02_calculate_bdp_activity.R --expression data/pprd/rice.tsv        --pairs results/01_candidates/rice.tsv        --uppercase-ids --output results/02_activity/rice.tsv
Rscript scripts/02_calculate_bdp_activity.R --expression data/pprd/maize.tsv       --pairs results/01_candidates/maize.tsv       --output results/02_activity/maize.tsv
Rscript scripts/02_calculate_bdp_activity.R --expression data/pprd/wheat.tsv       --pairs results/01_candidates/wheat.tsv       --output results/02_activity/wheat.tsv
```

Use `--gene-column Sample` if the first column in a PPRD export is named `Sample`.

## 3. Merge and filter candidates

Edit `config/species_manifest.example.tsv` or create an equivalent manifest.

```bash
Rscript scripts/03_01_merge_final_table.R --manifest config/species_manifest.example.tsv --output results/03_candidates/all_species.tsv
Rscript scripts/03_02_filter_clean_table.R --input results/03_candidates/all_species.tsv --max-distance 1000 --min-observations 1000 --output results/03_candidates/high_confidence.tsv
```

The final filter removes overlapping pairs, organellar/unplaced contigs, missing activity values, pairs longer than 1 kb, and pairs with fewer than 1,000 effective expression samples.

## 4. Extract BDP sequences

Edit `config/genomes.example.tsv` to point to uncompressed, indexed genome FASTA files.

```bash
python scripts/04_extract_bdp_sequences.py --candidates results/03_candidates/high_confidence.tsv --genomes config/genomes.example.tsv --lengths 128 256 512 1024 --output-dir results/04_sequences
```

Coordinates are converted from 1-based inclusive GFF3 coordinates to 0-based half-open FASTA slices.

## 5. Generate features

```bash
python scripts/05_01_evo2_features.py --fasta results/04_sequences/BDP_center_1024bp.fasta --layers 16 18 20 22 24 26 28 30 --output-dir results/05_features/evo2
python scripts/05_02_kmer_frequency.py --fasta results/04_sequences/BDP_center_1024bp.fasta --k 2 3 4 --output results/05_features/kmer_1024.pkl
python scripts/05_03_tfbs_identification.py --fasta results/04_sequences/BDP_center_1024bp.fasta --motifs data/motifs/JASPAR.meme --format meme --threshold 5 --output results/05_features/tfbs_1024.pkl
```

Evo2 features are mean-pooled hidden states. Extraction failures stop the run rather than inserting zero vectors. The TFBS score threshold is a log-odds cutoff and should be reported if changed.

## 6. Train and compare models

LightGBM evaluates top/bottom activity fractions from 0.01 to 0.14 and the manuscript learning-rate grid. Repeat `--feature NAME=PATH` to compare feature families in one result table.

```bash
python scripts/06_01_train_lightgbm.py --labels results/03_candidates/high_confidence.tsv --feature Evo2=results/05_features/evo2/BDP_center_1024bp_layer24.pkl --feature kmer=results/05_features/kmer_1024.pkl --feature TFBS=results/05_features/tfbs_1024.pkl --threads 8 --output results/06_models/lightgbm_summary.tsv
python scripts/06_02_sequence_cnn.py --labels results/03_candidates/high_confidence.tsv --fasta results/04_sequences/BDP_center_1024bp.fasta --label-ratio 0.10 --output results/06_models/cnn_summary.tsv
```

LightGBM defaults match the manuscript: `num_leaves=127`, `max_depth=9`, `min_child_samples=30`, L1/L2 regularization 1.0, feature fraction 0.7, bagging fraction 0.8, and bagging frequency 3. Metrics are AUROC, F1, and accuracy from stratified five-fold cross-validation.

## 7. Cross-species prediction

```bash
python scripts/07_01_single_species_transfer.py --labels results/03_candidates/high_confidence.tsv --features results/05_features/evo2/BDP_center_1024bp_layer24.pkl --output results/07_transfer/one_species.tsv
python scripts/07_02_two_species_transfer.py    --labels results/03_candidates/high_confidence.tsv --features results/05_features/evo2/BDP_center_1024bp_layer24.pkl --output results/07_transfer/two_species.tsv
python scripts/07_03_three_species_transfer.py  --labels results/03_candidates/high_confidence.tsv --features results/05_features/evo2/BDP_center_1024bp_layer24.pkl --output results/07_transfer/three_species.tsv
```

Each model is evaluated only on species absent from its training set.

## 8. SAE and known-regulatory analyses

The manuscript describes layer-24 Evo2 activations mapped from 4,096 to 32,768 SAE dimensions with top-k 64 sparsity. Override the dimensions or layer when using a different checkpoint.

```bash
python scripts/08_01_extract_sae_features.py --fasta results/04_sequences/BDP_center_1024bp.fasta --sae-checkpoint data/models/sae_layer24.pt --layer 24 --d-in 4096 --expansion-factor 8 --top-k 64 --output results/08_interpretation/sae.pkl
python scripts/08_04_sae_delta.py --sae results/08_interpretation/sae.pkl --labels results/03_candidates/high_confidence.tsv --fraction 0.10 --output results/08_interpretation/sae_delta.tsv
```

Generate TFBS position profiles from FIMO output, then calculate top-minus-bottom 10% deltas:

```bash
python scripts/08_02_aggregate_tfbs_profiles.py --fimo results/fimo/fimo.tsv --length 1024 --output results/08_interpretation/tfbs_profiles.tsv
python scripts/08_04_delta_profiles.py --profiles results/08_interpretation/tfbs_profiles.tsv --labels results/03_candidates/high_confidence.tsv --fraction 0.10 --output results/08_interpretation/tfbs_delta.tsv
```

NuPoP is run from a manifest because species and model codes are NuPoP-specific. The supplied example reproduces the provided Arabidopsis call (`species=10`, `model=4`). Convert NuPoP occupancy output to the same profile schema: `species`, `sequence_id`, `feature_id`, followed by position columns `1..L`.

```bash
Rscript scripts/08_03_nupop_profiles.R --manifest config/nupop.example.tsv --output-dir results/08_interpretation/nupop
python scripts/08_05_spearman_profiles.py --sae results/08_interpretation/sae_delta.tsv --regulatory results/08_interpretation/tfbs_delta.tsv --threshold 0.5 --output results/08_interpretation/sae_tfbs_spearman.tsv
```

Run the final command again with the NuPoP delta table to test SAE-nucleosome associations.

## Tests

```bash
python -m compileall scripts
pytest -q
```

The tests cover FASTA identifiers and balanced extreme-activity labels. Full Evo2, SAE, NuPoP, and model runs require the external datasets and weights described above.

________________________________________
#### 📝 Cite This Work
If you use this code or data, please cite our paper:  
Genomic language model-driven decoding of gene regulation: a case-study on predicting bidirectional promoter activity across plant species
[DOI: XXXXXXX]
________________________________________
#### 📧 Contact
For questions or collaboration requests, contact:  
**Huihui Li - lihuihui@caas.cn**
________________________________________
#### 🌟 Acknowledgments
•	Evo2 (Benegas et al., 2025) for the genomic language model.  
•	PPRD (Yu et al., 2022) for public RNA-seq data.  
•	LightGBM (Ke et al., 2017) for efficient gradient boosting.  
________________________________________
License: MIT License.
