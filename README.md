# GNN-BRAIN-CONNECTIVITY

**Graph Neural Networks for Predicting Neurological Disease from Brain Connectivity**

---

## 🧠 Overview

This project investigates whether **graph-based deep learning** can improve prediction of neurological and psychiatric conditions from brain connectivity data.

Traditional machine learning approaches treat brain connectivity matrices as flat vectors, ignoring the inherent **network structure** of the brain. In contrast, this project models the brain explicitly as a **graph**, where regions are nodes and connections are edges, and applies **Graph Neural Networks (GNNs)** to learn structure-aware representations.

The goal is to evaluate whether preserving **topological information** leads to better predictive performance and more meaningful representations of brain organization.

---

## 🎯 Research Motivation

Brain function emerges from interactions between distributed regions forming complex networks. This motivates the use of:

* **Connectomics** → modeling the brain as a network
* **Graph theory** → capturing topology and interactions
* **Graph Neural Networks** → learning directly from graph structure

Key question:

> Can graph-based models better capture disease-related alterations in brain connectivity than traditional approaches?

---

## 📊 Dataset

* **Source**: OpenNeuro
* **Dataset**: ds000030 (UCLA Neuropsychiatric Phenomics)
* **Modality**: Resting-state fMRI
* **Subjects used**: Subsample (≈5–15 subjects for rapid experimentation)
* **Labels**: Diagnostic categories (e.g., control vs condition)

### Data Characteristics

* BIDS-compliant structure
* Preprocessed or minimally processed fMRI time series
* Suitable for functional connectivity estimation

### Data Usage Strategy

To maintain efficiency and reproducibility:

* Only a subset of subjects is used
* Focus is on **methodological validation**, not dataset scale

---

## ⚙️ Methodology

### 1. Preprocessing & Time Series Extraction

Using **Nilearn**:

* Load fMRI volumes
* Apply atlas-based parcellation (e.g., AAL, Schaefer)
* Extract regional time series

---

### 2. Functional Connectivity Estimation

* Compute pairwise correlations between regions
* Generate **connectivity matrices (N × N)**
* Optionally:

  * Apply thresholding
  * Normalize edge weights

---

### 3. Graph Construction

Each subject is represented as a graph:

* **Nodes**: Brain regions (ROIs)
* **Edges**: Functional connectivity strength
* **Node Features**:

  * Identity matrix (baseline)
  * OR graph-derived features (degree, centrality)

This transforms neuroimaging data into a format suitable for **graph learning**.

---

### 4. Models

#### 🔹 Baseline: Multi-Layer Perceptron (MLP)

* Input: Flattened connectivity matrix
* Ignores graph structure
* Serves as comparison benchmark

---

#### 🔹 Graph Model: Graph Convolutional Network (GCN)

Implemented using **PyTorch Geometric**

**Architecture:**

* Graph Convolution layers (2–3)
* Nonlinear activation (ReLU)
* Global pooling (mean or max)
* Fully connected classification layer

**Objective:**
Learn embeddings that incorporate both:

* Node features
* Network topology

---

### 5. Optional Multimodal Extension

To simulate real-world neuroimaging pipelines:

* Add non-imaging features:

  * Age
  * Sex
* Concatenate with graph embeddings before classification

---

### 6. Training & Evaluation

* Train/test split or cross-validation
* Loss: Cross-entropy
* Metrics:

  * Accuracy
  * ROC-AUC

---

## 📈 Results

*(To be populated after experiments)*

Evaluation focuses on:

* Performance comparison: **GCN vs MLP**
* Impact of graph structure on prediction
* Stability across small sample sizes

---

## 🧪 Key Hypotheses

* Graph structure improves predictive performance
* Connectivity topology contains disease-relevant signals
* GNN embeddings capture biologically meaningful patterns

---

## 📁 Project Structure

```id="projstruct1"
GNN-BRAIN-CONNECTIVITY/
│
├── data/
│   ├── raw/                  # Downloaded OpenNeuro data
│   ├── processed/            # Connectivity matrices & graphs
│
├── notebooks/
│   ├── exploration.ipynb     # Data exploration & visualization
│
├── src/
│   ├── data_loader.py        # Load fMRI / connectivity data
│   ├── preprocess.py         # Time series extraction (Nilearn)
│   ├── connectivity.py       # Correlation matrix generation
│   ├── graph_builder.py      # Convert matrices → graphs
│   │
│   ├── models/
│   │   ├── gcn.py            # Graph Neural Network
│   │   ├── mlp.py            # Baseline model
│   │
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation metrics
│
├── results/
│   ├── figures/              # Plots
│   ├── metrics.json          # Results
│
├── config.yaml               # Experiment configuration
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash id="install1"
git clone https://github.com/YOUR_USERNAME/connectome-gnn.git
cd connectome-gnn

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## ☁️ S3 Selective Download (OpenNeuro Subset)

### 🧠 1. Set up AWS CLI (if not already)

```bash
pip install awscli
```

Check:

```bash
aws --version
```

### 📦 2. Create a local folder

```bash
mkdir ds000030_small
cd ds000030_small
```

### 🔥 3. Download ONLY 5 subjects (recommended method)

You can manually pick subjects, for example:

* sub-10159
* sub-10171
* sub-10189
* sub-10193
* sub-10201

### ⚡ Command (repeat for each subject)

Option A: full subject (safe but bigger)

```bash
aws s3 cp --no-sign-request \
s3://openneuro.org/ds000030/sub-10159/ \
./sub-10159/ \
--recursive
```

---

## ▶️ Usage

### 1. Preprocess data

```bash id="run1"
python src/preprocess.py
```

### 2. Generate connectivity matrices

```bash id="run2"
python src/connectivity.py
```

### 3. Train baseline model

```bash id="run3"
python src/train.py --model mlp
```

### 4. Train GNN model

```bash id="run4"
python src/train.py --model gcn
```

### 5. Evaluate

```bash id="run5"
python src/evaluate.py
```

---

## 🧠 Key Insights

* Brain networks are naturally suited for **graph-based modeling**
* Flattened representations discard **relational structure**
* GNNs enable **structure-aware learning** in neuroimaging

---

## ⚠️ Limitations

* Small sample size (subsampled dataset)
* Simplified preprocessing pipeline
* Basic GCN architecture (no attention mechanisms)
* Limited hyperparameter tuning

---

## 🔭 Future Work

* Incorporate **multimodal data** (clinical, behavioral)
* Explore advanced architectures:

  * Graph Attention Networks (GAT)
  * GraphSAGE
* Apply to **longitudinal disease progression**
* Integrate with:

  * BIDS pipelines
  * DataLad workflows
* Scale to larger datasets

---

## 🔁 Reproducibility

* Structured project layout
* Version-controlled code
* Configurable experiments
* Designed with **FAIR principles** in mind

---

## 📚 References

* OpenNeuro dataset (ds000030)
* PyTorch Geometric documentation
* Nilearn documentation

---

## 👤 Author

Patrick Filima
Computational Neuroscience | Neuroinformatics | AI for Brain Connectivity

---

## 📬 Contact

[filimapatrick@gmail.com](mailto:filimapatrick@gmail.com)

---

## 💡 Project Note

This project was developed as a focused exploration of **graph-based learning for connectomics**, with an emphasis on clarity, reproducibility, and alignment with current research directions in computational neuroscience.
# GNN-BRAIN-CONNECTIVITY
