# Self-Assembly-Score-Guided Flow Matching for Drug-Pair BRAID Pharmacodynamic Parameter Generation and Herbal Compatibility Discovery

This repository contains the code and data for self-assembly score prediction, Conditional Flow Matching-based BRAID pharmacodynamic parameter generation, and molecular-to-herbal compatibility prediction.

## 1. Repository Structure

```text
.
├── flow_model_train_data.csv
├── flow_model_trans.py
├── flow_model_use.py
├── herb_prediction.py
├── self_assembly_data.pt
├── self_assembly_data.tsv
├── self_assembly_models.py
├── self_assembly_train.py
└── tsv_pt_data.py
```

---

## 2. File Description

### 2.1 Flow Matching Module

#### `flow_model_train_data.csv`

Training data for the Flow Matching model.

The dataset contains the molecular-pair information, cell-line conditions, self-assembly scores, and corresponding BRAID pharmacodynamic parameters used for model training.

This file is used by:

```text
flow_model_trans.py
```

---

#### `flow_model_trans.py`

Training script for the Conditional Flow Matching model.

The script uses `flow_model_train_data.csv` to train the Flow Matching model.

The model takes molecular-pair information, cell-line information, and the self-assembly score as conditional information and learns to generate the corresponding seven-dimensional BRAID parameter vector:

```text
E0
Ef
logEC50_A
logEC50_B
mA
mB
κ
```

After training, the model can be used for molecular-level BRAID parameter prediction.

---

#### `flow_model_use.py`

Inference and usage example for the trained Flow Matching model.

This script demonstrates how a specific molecular pair is processed together with:

```text
Molecular pair
Cell line
Self-assembly score
```

to generate the corresponding BRAID pharmacodynamic parameters.

The script can be used as an example for applying the trained Flow Matching model to new molecular pairs.

---

### 2.2 Herbal Compatibility Module

#### `herb_prediction.py`

Script for transferring molecular-level predictions to the herbal medicine level.

The script uses molecular prediction results together with compound–herb relationships to evaluate the compatibility of herbal pairs.

The general process is:

```text
Molecular pairs
      ↓
Molecular-level prediction
      ↓
Compound–herb mapping
      ↓
Herbal-level aggregation
      ↓
Herbal compatibility prediction
```

The output can be used to rank candidate herbal pairs for further analysis or experimental validation.

---

### 2.3 Self-Assembly Prediction Module

#### `self_assembly_data.tsv`

Original tabular training dataset for the self-assembly prediction model.

This file contains the collected molecular self-assembly training data before preprocessing.

It is used as the input of:

```text
tsv_pt_data.py
```

---

#### `tsv_pt_data.py`

Preprocessing script for the self-assembly training data.

The script converts the original tabular dataset:

```text
self_assembly_data.tsv
```

into a processed and vectorized PyTorch dataset:

```text
self_assembly_data.pt
```

The generated `.pt` file is then used for training the self-assembly prediction model.

---

#### `self_assembly_data.pt`

Processed and vectorized self-assembly training dataset.

This file is generated from:

```text
self_assembly_data.tsv
```

using:

```text
tsv_pt_data.py
```

It is used directly by:

```text
self_assembly_train.py
```

for self-assembly model training.

> `self_assembly_data.pt` is the processed training dataset, not the model architecture.

---

#### `self_assembly_models.py`

Model definition for the self-assembly prediction network.

This file contains the neural network architecture used by:

```text
self_assembly_train.py
```

---

#### `self_assembly_train.py`

Training script for the self-assembly prediction model.

The script loads the processed dataset:

```text
self_assembly_data.pt
```

and uses the model architecture defined in:

```text
self_assembly_models.py
```

to train the self-assembly prediction model.

After training, the model can be used to predict the self-assembly score of molecular pairs.

The predicted self-assembly score is subsequently used as a condition in the Flow Matching model.

---

## 3. Recommended Usage

The complete workflow is executed in the following order.

### Step 1. Preprocess the self-assembly data

Run:

```bash
python tsv_pt_data.py
```

Input:

```text
self_assembly_data.tsv
```

Output:

```text
self_assembly_data.pt
```

---

### Step 2. Train the self-assembly prediction model

Run:

```bash
python self_assembly_train.py
```

The script uses:

```text
self_assembly_data.pt
self_assembly_models.py
```

to train the self-assembly prediction model.

The trained model produces a self-assembly score for a given molecular pair.

---

### Step 3. Train the Flow Matching model

Run:

```bash
python flow_model_trans.py
```

The script uses:

```text
flow_model_train_data.csv
```

as the training dataset.

The trained Flow Matching model learns to generate BRAID pharmacodynamic parameters conditioned on molecular-pair information, cell-line information, and self-assembly score.

---

### Step 4. Perform molecular-level inference

Run:

```bash
python flow_model_use.py
```

This script provides an example of using the trained Flow Matching model to generate BRAID parameters for a specific molecular pair.

The generated parameters are:

```text
E0
Ef
logEC50_A
logEC50_B
mA
mB
κ
```

---

### Step 5. Perform herbal compatibility prediction

Run:

```bash
python herb_prediction.py
```

This script transfers molecular-level predictions to the herbal level and generates herbal compatibility predictions and candidate rankings.

---

## 4. File Dependency

The relationship between the files can be summarized as follows:

```text
                    Self-Assembly Module
                            
self_assembly_data.tsv
          │
          ▼
   tsv_pt_data.py
          │
          ▼
self_assembly_data.pt
          │
          ├───────────────┐
          ▼               │
self_assembly_train.py    │
          │               │
          ▼               │
self_assembly_models.py   │
          │               │
          ▼               │
 Self-Assembly Model      │
          │               │
          ▼               │
 Self-Assembly Score      │
          │               │
          └───────┬───────┘
                  │
                  ▼
        Flow Matching Module

flow_model_train_data.csv
          │
          ▼
flow_model_trans.py
          │
          ▼
 Conditional Flow Matching
          │
          ▼
 BRAID Parameters
          │
          ▼
flow_model_use.py
          │
          ▼
Molecular-level Prediction
          │
          ▼
   herb_prediction.py
          │
          ▼
Herbal Compatibility Prediction
```

---

## 5. Summary of Each File

| File | Type | Main Function |
|---|---|---|
| `flow_model_train_data.csv` | Dataset | Training data for Flow Matching |
| `flow_model_trans.py` | Training | Train the Conditional Flow Matching model |
| `flow_model_use.py` | Inference | Generate BRAID parameters for a molecular pair |
| `herb_prediction.py` | Application | Map molecular predictions to herbal compatibility |
| `self_assembly_data.tsv` | Dataset | Original self-assembly training data |
| `tsv_pt_data.py` | Preprocessing | Convert `.tsv` data into vectorized `.pt` data |
| `self_assembly_data.pt` | Processed Dataset | Vectorized self-assembly training data |
| `self_assembly_models.py` | Model | Self-assembly model architecture |
| `self_assembly_train.py` | Training | Train the self-assembly prediction model |

---

## 6. Overall Workflow

The complete computational workflow is:

```text
Self-Assembly Training Data
            ↓
    Data Preprocessing
            ↓
 Self-Assembly Prediction
            ↓
   Self-Assembly Score
            ↓
 Conditional Flow Matching
            ↓
 BRAID Parameter Generation
            ↓
 Molecular-Level Prediction
            ↓
 Molecular-to-Herbal Mapping
            ↓
Herbal Compatibility Prediction
            ↓
 Candidate Herbal Pairs
```

The repository is organized so that the self-assembly prediction, Flow Matching model training, molecular-level inference, and herbal compatibility prediction can be reproduced as separate steps.