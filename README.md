### BDP-Identifier: Genomic Language Model-Based Prediction of Bidirectional Promoter Activity
This repository contains the code and data associated with the study "Genomic language model-driven decoding of gene regulation: a case-study on predicting bidirectional promoter activity across plant species" (https://github.com/AIBreeding/BDP-identifier). The workflow leverages genomic language models (gLMs) and machine learning (ML) to predict bidirectional promoter activity (BDP activity) from DNA sequences in plant species.
________________________________________
#### 📁 Directory Structure
```bash
BDP-Identifier/  
├── 01.generate_input_sequence.py        # Generate input DNA sequences for BDP candidates  
├── 02.generate_embedding.py             # Generate Evo2 embeddings for DNA sequences  
├── 03.grid_search_emb_semiML.py         # Grid search for optimal ML model parameters using embeddings  
├── 04.emb_ML.py                         # Train ML models using Evo2 embeddings  
├── 05.freq_ML.py                        # Train ML models using di-nucleotide frequency features  
├── BDP_candidates.csv                   # Dataset of candidate BDPs from Arabidopsis, rice, maize, and wheat  
└── README.md                            # This file
```  
________________________________________
#### 🧪 Overview of the Workflow
•	Identify BDP Candidates:  
&emsp;&emsp; •	Extract DNA sequences flanking bidirectional promoters (≤1,000 bp) from plant genomes (Arabidopsis, rice, maize, wheat).  
&emsp;&emsp; •	Quantify BDP activity using Pearson correlation coefficients (PCC) from RNA-seq data.  
•	Feature Generation:  
&emsp;&emsp; •	Evo2 Embeddings: Use the pre-trained genomic language model Evo2 to generate position-specific embeddings.  
&emsp;&emsp; •	Di-Nucleotide Frequencies: Compute di-nucleotide frequencies from six genomic regions (CDS, UTRs, introns, etc.).  
•	Model Training:  
&emsp;&emsp; •	Train LightGBM models to predict BDP activity using embeddings or di-nucleotide frequency features.  
&emsp;&emsp; •	Evaluate performance using AUROC, Accuracy, and F1-score.  
•	Cross-Species Generalization:  
&emsp;&emsp; •	Test models trained on 1-3 species to predict BDP activity in a held-out species (e.g., Spartina alterniflora).  
•	Experimental Validation:  
&emsp;&emsp; •	Validate predicted high-activity BDPs using transient dual-reporter assays in Nicotiana benthamiana.  
________________________________________
#### 🛠️ Installation & Requirements
•	Python Environment:  
&emsp;&emsp; •	Python 3.8+  
&emsp;&emsp; •	Required packages: numpy, pandas, scikit-learn, lightgbm, torch, biopython  
•	Evo2 Model:  
&emsp;&emsp; •	Download the pre-trained Evo2_7B model (https://github.com/your-repo/evo2-models).  
________________________________________
#### 🚀 Usage
•	Generate Input Sequences:  
```bash
python 01.generate_input_sequence.py --species Arabidopsis --output_dir data/sequences  
```
•	Generate Evo2 Embeddings:  
```bash
python 02.generate_embedding.py --input_file data/sequences/Arabidopsis.bed --output_file data/embeddings/Arabidopsis.npy
```
•	Train ML Models:
&emsp; •	Using embeddings:  
```bash
python 04.emb_ML.py --embedding_file data/embeddings/Arabidopsis.npy --label_file data/labels/Arabidopsis.csv  
```
&emsp; •	Using di-nucleotide frequencies:
```bash
python 05.freq_ML.py --frequency_file data/frequencies/Arabidopsis.csv --label_file data/labels/Arabidopsis.csv  
```
________________________________________
#### 📊 Data Description
•	BDP_candidates.csv:  
&emsp;&emsp; •	Contains 4,524 high-confidence BDP candidates from Arabidopsis, rice, maize, and wheat.  
•	Public RNA-seq Data:  
&emsp;&emsp; •	Expression data from the Plant Public RNA-seq Database (PPRD) (Yu et al., 2022).   
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
