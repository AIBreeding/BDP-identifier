# BDP-identifier

Identifying plant bidirectional promoter (BDP) candidates, quantifying their activity, generating sequence features, training prediction models, and relating Evo2 sparse-autoencoder (SAE) features to known regulatory signals.

## What this workflow does

This pipeline starts from a genome assembly, a GFF3 gene annotation, and public RNA-seq expression profiles. It produces a reproducible set of candidate bidirectional promoters (BDPs), assigns each candidate a quantitative activity proxy, extracts centered promoter sequences, constructs multiple sequence representations, and evaluates how well BDP activity can be predicted within and across species.

The main outputs are:

- genomic coordinates of adjacent divergent gene pairs and a filtered high-confidence BDP set;
- `cor_log`, the Pearson correlation between log10-transformed expression profiles of the two flanking genes, used as the BDP activity proxy;
- 128-, 256-, 512-, and 1,024-bp BDP-centered DNA sequences;
- Evo2 embeddings, k-mer frequencies, TFBS features, and raw-sequence CNN inputs;
- LightGBM/CNN benchmarking results and one-, two-, and three-species transfer results; and
- position-resolved Evo2-SAE, TFBS, and nucleosome-affinity delta profiles for comparing high- and low-activity BDPs.

## Data to download

The manuscript uses genome assemblies and gene annotations from **Ensembl Plants release 55** and transcriptome data from the **Plant Public RNA-seq Database (PPRD)**.

| Data | Species | Source and download location | Files needed |
|---|---|---|---|
| Genome assembly | *A. thaliana*, *O. sativa*, *Z. mays*, *T. aestivum* | [Ensembl Plants release 55 FTP](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/fasta/) | One primary/toplevel DNA FASTA per species (`*.dna.toplevel.fa.gz` or equivalent) |
| Gene annotation | Same four species | [Ensembl Plants release 55 GFF3 FTP](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/gff3/) | One matching GFF3 annotation per species (`*.gff3.gz`) |
| Public RNA-seq expression | Same four species | [Plant Public RNA-seq Database (PPRD)](http://116.205.136.219/) | Gene-by-sample expression matrix for each species |

Species-specific Ensembl Plants release 55 directories:

- [Arabidopsis thaliana FASTA](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/fasta/arabidopsis_thaliana/dna/) and [GFF3](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/gff3/arabidopsis_thaliana/)
- [Oryza sativa FASTA](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/fasta/oryza_sativa/dna/) and [GFF3](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/gff3/oryza_sativa/)
- [Zea mays FASTA](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/fasta/zea_mays/dna/) and [GFF3](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/gff3/zea_mays/)
- [Triticum aestivum FASTA](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/fasta/triticum_aestivum/dna/) and [GFF3](https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-55/gff3/triticum_aestivum/)

The manuscript reports 28,164 Arabidopsis, 19,664 maize, 11,726 rice, and 5,816 wheat RNA-seq samples in the full PPRD datasets. Download the complete species expression matrices rather than tissue-restricted subsets. Before step 2, arrange each matrix as one gene per row and one RNA-seq sample per column, with the gene identifier in `gene_id` (or pass its actual name with `--gene-column`). Gene identifiers must match the corresponding GFF3 annotation; the provided command uppercases rice identifiers to reproduce the source analysis.

Suggested local layout:

```text
data/
  genomes/       # four genome FASTA files
  gff/           # four matching GFF3 files
  pprd/          # four gene-by-sample expression tables
  motifs/        # JASPAR plant motif file for TFBS analysis
  models/        # Evo2 and SAE weights (not tracked by Git)
```

The FTP and PPRD URLs above are the source locations stated or implied by the manuscript. Archive mirrors and filenames can change; retain the release number and record the exact downloaded filenames/checksums for reproducibility.

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

## Data and code availability

As stated in the manuscript, transcriptome datasets used to calculate BDP activity were obtained from PPRD. Genome assemblies and gene annotations for the four focal species were obtained from Ensembl Plants release 55. Processed high-confidence BDP candidate tables should be deposited with this repository. The manuscript identifies the code repository as [AIBreeding/BDP-identifier](https://github.com/AIBreeding/BDP-identifier).

## Citation

Please cite the accompanying manuscript when using this workflow:

> *Genomic Language Models Decode Bidirectional Promoter Activity and Reveal Conserved Regulatory Patterns Across Plant Species.* Manuscript in preparation.

The supplied Word document does not contain a complete author list or bibliographic publication details. Replace the citation above with the final author list, journal, year, DOI, and version before public release.
________________________________________
## 📧 Contact
For questions or collaboration requests, contact:  
**Huihui Li - lihuihui@caas.cn**
________________________________________
License: MIT License.
