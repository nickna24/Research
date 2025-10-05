# Example Titanic ML Pipeline with HTCondor DAGMan

This project demonstrates how to structure a **machine learning workflow** using **HTCondor DAGMan**, with the [Titanic dataset](https://www.kaggle.com/c/titanic).  
It serves as a lightweight skeleton for modular ML pipelines and shows how to chain preprocessing, training, and evaluation jobs via DAG dependencies. For the sake of demonstration, I'm using a 3-layer feedforward neural network (MLP). 

---

## Pipeline Overview

| Stage | Script | Purpose | Output |
|-------|---------|----------|---------|
| **Preprocess** | `scripts/preprocess_data.py` | Cleans raw CSVs, encodes features, saves `.npy` arrays + metadata | `data/preprocessed/*` |
| **Train** | `scripts/train_model.py` | Trains a small MLP with variable hyperparameters | `checkpoints/model_*.pt` |
| **Evaluate** | `scripts/evaluate_model.py` | Loads a checkpoint and computes validation metrics | `logs/metrics_*.txt` |

Each stage runs as its own Condor job, chained by DAGMan.

---

## Directory Structure

```pgsql
titanic_dagman_project/
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── preprocessed/
│ ├───── train_proc.npy
│ ├───── test_proc.npy
│ ├───── labels.npy
│ └───── meta.json
├── scripts/
│ ├── preprocess_data.py
│ ├── train_model.py
│ └── evaluate_model.py
├── submit/
│ ├── preprocess.sub
│ ├── train.sub
│ └── eval.sub
├── checkpoints/
├── logs/
├── out/
├── err/
└── workflow.dag
```

## DAG Workflow

### `workflow.dag`
```dag
# --- Stage 1: Preprocess ---
JOB PREPROC submit/preprocess.sub

# --- Stage 2: Train ---
JOB TRAIN1 submit/train.sub
VARS TRAIN1 OPT="adam" LR="0.001" WPRED="1.0" WCONST="0.0005"

JOB TRAIN2 submit/train.sub
VARS TRAIN2 OPT="sgd" LR="0.01" WPRED="1.0" WCONST="0.0005"

PARENT PREPROC CHILD TRAIN1 TRAIN2

# --- Stage 3: Evaluate ---
JOB EVAL1 submit/eval.sub
VARS EVAL1 OPT="adam" LR="0.001"

JOB EVAL2 submit/eval.sub
VARS EVAL2 OPT="sgd" LR="0.01"

PARENT TRAIN1 CHILD EVAL1
PARENT TRAIN2 CHILD EVAL2
Workflow Summary:
PREPROC → TRAIN(Adam, SGD) → EVAL(Adam, SGD)
```

###  Model & Hyperparameters
A 3-layer MLP for binary classification (Survived vs Not Survived):

```python
nn.Linear(in_dim, 64) → ReLU → nn.Linear(64, 32) → ReLU → nn.Linear(32, 1)
```
### Arguments passed from Condor submit files:

| Argument | Description | Example |
|-----------|--------------|----------|
| `--optimizer` | Optimizer type | `adam`, `sgd` |
| `--lr` | Learning rate | `0.001`, `0.01` |
| `--w_pred` | Weight for BCE loss | `1.0` |
| `--w_const` | Weight for L2 regularization | `0.0005` |
| `--checkpoint_path` | Path to save model | `checkpoints/model_adam_0.001.pt` |

### Checkpoints
Each training job outputs:


```php-template
checkpoints/model_<OPT>_<LR>.pt
```
Each checkpoint stores:

```python
{
  "state_dict": model weights,
  "meta": {
    "optimizer": "adam",
    "lr": 0.001,
    "w_pred": 1.0,
    "w_const": 0.0005,
    "feature_cols": [...],
    "num_means": {...},
    "num_stds": {...},
    "cat_domains": {...},
    "val_indices": [...],
    "best_val_acc": 0.89
  }
}
```

Evaluation jobs read this file to compute metrics.

### Evaluation Outputs
Each EVAL job produces:

| File | Location | Description |
|------|-----------|--------------|
| `metrics_$(OPT)_$(LR).txt` | `logs/` | Validation loss & accuracy |
| `submission.csv` | `logs/` | Optional Kaggle-style predictions |
| `.out`, `.err`, `.log` | `out/`, `err/`, `logs/` | Condor job outputs |

Example metric file:

```makefile
val_loss: 0.451623
val_acc: 0.892857
```

### GPU Configuration
Both training and evaluation jobs request one GPU:

```sub
request_gpus = 1
requirements = (CUDADeviceName =!= UNDEFINED)
```

Each job uses:

1 CPU core

1 GPU

2 GB RAM


### Running the Workflow
From the project root:

```bash
condor_submit_dag workflow.dag
```

Monitor progress:

```bash
condor_q
condor_q -dag
```

Artifacts are written to:

logs/ — logs, metrics, submission

checkpoints/ — trained model files

out/ & err/ — stdout/stderr for Condor jobs

### Artifact Flow
Stage	Input	Output
Preprocess	Raw CSVs	.npy files + meta.json
Train	Preprocessed data	.pt checkpoint
Evaluate	Checkpoint	Metrics + submission CSV

### Example Output Tree
```pgsql
checkpoints/
├── model_adam_0.001.pt
├── model_sgd_0.01.pt

logs/
├── preprocess.log
├── train_adam_0.001.log
├── eval_adam_0.001.log
├── metrics_adam_0.001.txt
├── metrics_sgd_0.01.txt
└── submission.csv

out/
├── train_adam_0.001.out
└── eval_adam_0.001.out

err/
├── train_adam_0.001.err
└── eval_adam_0.001.err
```

### Extending the Pipeline
Add sweeps: create new TRAIN + EVAL pairs in workflow.dag.

GANs: split generator/discriminator training into separate DAG nodes.

Autonomous vehicle ML: replace preprocess_data.py with a script that handles sensor or image data.

Metrics aggregation: add a post-evaluation DAG stage to parse and summarize metrics automatically.
