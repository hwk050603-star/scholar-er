# Soft-Aligned Attentive Neighborhood Injection for Heterogeneous Scholar Entity Resolution

![python](https://img.shields.io/badge/python-3.9--3.11-blue)
![pytorch](https://img.shields.io/badge/pytorch-%3E%3D2.1-brightgreen)
![transformers](https://img.shields.io/badge/transformers-%3E%3D4.35-orange)

This repository contains the source code, processed data, checkpoints, baseline-format data, and analysis scripts for the paper **"Soft-Aligned Attentive Neighborhood Injection for Heterogeneous Scholar Entity Resolution"**.

Scholar entity resolution is difficult when records come from heterogeneous sources: names may be written in Chinese, English, abbreviation, or reversed order; affiliations may shift over time; and important fields such as research interests may be missing from one side. Our method introduces **Soft-Aligned Attentive Neighborhood Injection (SANI)**, which retrieves neighborhood scholars, softly aligns their contextual signals to the target scholar representation, and injects the aggregated neighborhood evidence into a multilingual encoder for match/non-match prediction.

<img src="figure/pipeline.pdf" width="900" />

## Code Structure

```sh
|-- blocking/                       # Blocking, augmented entity tables, candidate pairs, train/valid/test splits
|-- pretrain/                       # Neighbor search scripts and processed pre-training JSONL files
|   |-- dataset/
|-- pretrain_data_builders/         # Builders for HPC, MFP, and TAM pre-training tasks
|-- challenging_cases/              # Robustness test sets and evaluation script
|-- baseline_data/                  # Converted data for BatchER, ComEM, Ditto, HierGAT, and Sudowoodo
|-- case_study/                     # Case-study pairs, extracted neighbor weights, and heatmaps
|-- figure/                         # Paper figures and analysis plots
|-- vendor_src/                     # Vendored retriv 0.2.3 sparse retriever
|-- model.py                        # SANI layer and scholar matching models
|-- pretrain.py                     # Multi-task SER-oriented pre-training
|-- finetune.py                     # Fine-tuning for scholar pair classification
|-- prepare_pretraining_dataset.py  # Unified entry for building HPC/MFP/TAM data
|-- plot_case_heatmap.py            # Case-study attention heatmap generator
|-- top-k.py                        # Neighborhood-size K analysis plot
|-- requirements.txt
```

## Method Overview

SANI first serializes heterogeneous scholar attributes into pseudo-sentences with schema tokens `COL` and `VAL`, then encodes scholar pairs with XLM-R. For each scholar, retrieved neighbors are encoded with the same language model. The target scholar span acts as a query, while neighbor representations provide keys and values. A soft attention module computes relevance weights over neighbors, aggregates useful neighborhood evidence, and injects it back into the scholar representation through residual fusion and self-attention.

<img src="figure/SANI.pdf" width="900" />

The training pipeline has two stages:

1. **SER-oriented pre-training** with three auxiliary tasks:
   - **HPC**: Homonym Pair Classification.
   - **MFP**: Masked Field Prediction.
   - **TAM**: Timeline-Aware Affiliation Matching.
2. **Fine-tuning** on labeled heterogeneous scholar pairs for binary match prediction.

The following example illustrates why neighborhood evidence is useful: missing or ambiguous fields can be inferred from structurally related scholars in the same academic context.

<img src="figure/Sample.pdf" width="900" />

## Datasets

The processed data used by this repository is already organized under `blocking/` and `pretrain/dataset/`.

### Entity Tables and Pair Splits

| File                                   | Description                      |        Size |
| :------------------------------------- | :------------------------------- | ----------: |
| `blocking/dataset_A_aug.csv`           | Augmented source-A scholar table | 18,572 rows |
| `blocking/dataset_B_aug.csv`           | Augmented source-B scholar table | 13,727 rows |
| `blocking/A_B_mapping.csv`             | Gold entity mappings             | 12,112 rows |
| `blocking/A_B_blocking_candidates.csv` | Retrieved candidate pairs        | 92,860 rows |
| `blocking/train.txt`                   | Training pairs                   | 20,000 rows |
| `blocking/valid.txt`                   | Validation pairs                 |  5,000 rows |
| `blocking/test.txt`                    | Test pairs                       |  5,000 rows |

Each pair file is a tab-separated file in the format:

```text
record_left    record_right    label
```

where `label=1` means match and `label=0` means non-match.

### Pre-training Data

| File                                        | Task                                |             Size |
| :------------------------------------------ | :---------------------------------- | ---------------: |
| `pretrain/dataset/hpc_dataset.jsonl`        | Homonym Pair Classification         |   72,672 samples |
| `pretrain/dataset/mfp_dataset.jsonl`        | Masked Field Prediction             |   18,103 samples |
| `pretrain/dataset/tam_dataset.jsonl`        | Timeline-Aware Affiliation Matching |   39,825 samples |
| `pretrain/dataset/dataset_A_neighbors.json` | Source-A neighbor lists             | top-10 neighbors |
| `pretrain/dataset/dataset_B_neighbors.json` | Source-B neighbor lists             | top-10 neighbors |

### Challenging Cases

The directory `challenging_cases/` contains three stress-test subsets:

| Directory             | Purpose                                                |
| :-------------------- | :----------------------------------------------------- |
| `name_ambiguity/`     | Same or highly similar names with different identities |
| `missing_attributes/` | Records with missing or incomplete fields              |
| `affiliation_shifts/` | Scholars whose affiliations change across time         |

## Model Checkpoint

The model checkpoint is available from an anonymized Hugging Face repository for review:

```text
https://huggingface.co/anonymous-er-artifact/scholar-er-review
```

The final non-anonymous checkpoint repository will be released upon acceptance.

## Quick Start

### Step 1: Environment Setup

Create a Python environment and install dependencies:

```bash
conda create -n sani-ser python=3.10
conda activate sani-ser
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build matching your CUDA version before installing the remaining dependencies. The code uses `xlm-roberta-base` by default. If a local `xlm-roberta-base/` directory exists in the project root, it will be used automatically; otherwise Hugging Face Transformers will download the model.

### Step 2: Pre-training

Run SER-oriented multi-task pre-training with HPC, MFP, and TAM:

```bash
python pretrain.py \
  --hpc_jsonl pretrain/dataset/hpc_dataset.jsonl \
  --mfp_jsonl pretrain/dataset/mfp_dataset.jsonl \
  --tam_jsonl pretrain/dataset/tam_dataset.jsonl \
  --dataset_A_neighbors pretrain/dataset/dataset_A_neighbors.json \
  --dataset_B_neighbors pretrain/dataset/dataset_B_neighbors.json \
  --save_dir checkpoints \
  --use_sani 1 \
  --aggregation_method attention \
  --device 0 \
  --max_epochs 2 \
  --batch_size 32 \
  --lr 1e-5
```

The pre-training checkpoint is saved to:

```text
checkpoints/model.pth
```

### Step 3: Fine-tuning

Fine-tune SANI on the labeled scholar pair classification task:

```bash
python finetune.py \
  --train_path blocking/train.txt \
  --dev_path blocking/valid.txt \
  --test_path blocking/test.txt \
  --dataset_A blocking/dataset_A_aug.csv \
  --dataset_B blocking/dataset_B_aug.csv \
  --dataset_A_neighbors pretrain/dataset/dataset_A_neighbors.json \
  --dataset_B_neighbors pretrain/dataset/dataset_B_neighbors.json \
  --pretrained_ckpt checkpoints/model.pth \
  --save_dir finetune_checkpoints \
  --use_sani 1 \
  --aggregation_method attention \
  --device 0 \
  --epochs 10 \
  --batch_size 32 \
  --lr 3e-5
```

The best fine-tuned model is saved to:

```text
finetune_checkpoints/model.pth
```

### Step 4: Evaluation

Evaluate a fine-tuned checkpoint on any challenging subset:

```bash
python challenging_cases/evaluate_challenging.py \
  --ckpt finetune_checkpoints/model.pth \
  --test_path challenging_cases/name_ambiguity/test.txt \
  --output challenging_cases/name_ambiguity/result.json \
  --device 0
```

You can replace `name_ambiguity` with `missing_attributes` or `affiliation_shifts`.

## Analysis and Case Study

### Neighborhood Size K

The supplied K analysis shows that using a moderate number of neighbors works best in this setup. In the saved plot, K=3 reaches the highest F1 score.

<img src="figure/k_analysis_prf.pdf" width="560" />

Regenerate the plot with:

```bash
python top-k.py
```

### Neighbor Attention Visualization

To extract neighbor relevance weights and draw the case-study heatmap:

```bash
python case_study/case_study_neighbor_weights.py \
  --ckpt finetune_checkpoints/model.pth

python plot_case_heatmap.py
```

The generated files are:

```text
case_study/case_neighbor_weights.csv
case_study/case_neighbor_heatmap.png
case_study/case_neighbor_heatmap.pdf
```

<img src="case_study/case_neighbor_heatmap.pdf" width="620" />

## Baseline Data

Converted files for several baseline systems are provided in `baseline_data/`:

| Directory                  | Format                                            |
| :------------------------- | :------------------------------------------------ |
| `baseline_data/BatchER/`   | JSON train/valid/test and challenging-case files  |
| `baseline_data/ComEM/`     | CSV files for pair matching and challenging cases |
| `baseline_data/Ditto/`     | Tab-separated pair files                          |
| `baseline_data/HierGAT/`   | HierGAT-style train/valid/test files              |
| `baseline_data/Sudowoodo/` | Supervised and unlabeled pair files               |
