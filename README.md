# SenseWhy
# SenseWhy: Overeating Behavior Analysis

This repository contains the analysis pipeline for the **SenseWhy study**, focusing on predicting, clustering, and visualizing patterns of **overeating behavior** using **sensor features** and **EMA (Ecological Momentary Assessment)** data.

The workflow integrates:
1. **Classification models** (XGBoost, SVM, Naive Bayes, MLP) to predict overeating events.  
2. **Representation learning + clustering** (MLP hidden layers → UMAP → KMeans/DBSCAN) to identify latent overeating clusters.  
3. **Post-hoc analysis (z-score polar plots)** to interpret principal indicators driving overeating clusters.

---

## Repository Structure


SenseWhy/
│
├── classification/                # Predictive modeling pipeline
│   ├── overeating_sensor_ema_model_eval.ipynb
│   └── 
│
├── clustering/                    # Representation learning + UMAP + clustering
│   ├── sensewhy_overeating_umap_clusters.ipynb
│   └── 
│
├── posthoc/                       # Z-score analysis & polar plots
│   ├── overeating_clusters_zscore_posthoc_analysis.ipynb
│   └── 
│
│
└── README.md                      # This file

---

## 1. Classification (Prediction of Overeating)

**Notebook:** `classification/overeating_sensor_ema_model_eval.ipynb`

- **Input:**  
  - `overeating_sensor_features.csv`  
  - `overeating_ema_features.csv`  

- **Models implemented:**  
  - XGBoost (Bayesian optimization for hyperparameters)  
  - Support Vector Machine (GridSearchCV tuned)  
  - Logistic Regression, Naive Bayes (baseline)  

- **Evaluation metrics:**  
  - 5-fold Stratified CV  
  - ROC (AUROC ± SD)  
  - Precision-Recall (AUPRC ± SD)  
  - Calibration curve + Brier score  

- **Outputs:**  
  - Comparative plots of classifiers (`result/val_test_*.pdf/png`)  

---

## 2. Representation Learning & Clustering

**Notebook:** `clustering/sensewhy_overeating_umap_clusters.ipynb`

- **Steps:**  
  1. Train an MLP on EMA features with 10-fold CV (hidden layers: `(200, 100, 50, 25, 5)`).
  2. Extract hidden representations from the penultimate layer (`Dim_1`–`Dim_5`).
  3. Project embeddings into 2D using **UMAP**.
  4. Cluster UMAP embeddings using **KMeans** (default: 30 clusters).  
     - Cluster labels stored in `res_df_nnout_umap2d_clustered.csv`.  

- **Outputs:**  
  - 2D UMAP embeddings colored by cluster.  
  - Clustered DataFrame with overeating labels + cluster IDs.  

---

## 3. Post-hoc Z-Score Analysis (Polar Plot)

**Notebook:** `posthoc/overeating_clusters_zscore_posthoc_analysis.ipynb`

- **Objective:**  
  Interpret overeating clusters using **z-scores of EMA features** to identify **principal indicators**.

- **Steps:**  
  1. Compute z-scores for each feature across clusters.  
  2. Identify features with |Z-score| ≥ 1.  
  3. Visualize per-cluster feature profiles using **pyCirclize** polar plots.  
  4. Add annotations & thresholds (Z = ±1).  

- **Outputs:**  
  - `polarplot_updated_10_21.pdf` → circos-style visualization of cluster-specific principal indicators.  

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/SenseWhy.git
cd SenseWhy

pip install -r requirements.txt


## Requirements

This project was developed and tested on **Python 3.9+**.  

Install dependencies with:

```bash
pip install -r requirements.txt
Core dependencies
pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

xgboost

shap

bayesian-optimization

umap-learn

pyCirclize

joblib

Usage
Run classification models (XGBoost, SVM, NB, Logistic Regression):

bash
Copy
Edit
jupyter notebook classification/overeating_sensor_ema_model_eval.ipynb
Run UMAP + clustering (MLP hidden representations → UMAP → KMeans):

bash
Copy
Edit
jupyter notebook clustering/sensewhy_overeating_umap_clusters.ipynb
Run post-hoc z-score analysis (polar plot of principal indicators):

bash
Copy
Edit
jupyter notebook posthoc/overeating_clusters_zscore_posthoc_analysis.ipynb
How to Cite
If you use this code, please cite both the SenseWhy paper and the underlying tools/packages.

Cite the Paper
Shahabi F, Wei B, Romano C, McCloskey R, Lin AW, Pedram M, Schauer J, Stump T, Alshurafa N.
Unveiling overeating patterns within digital longitudinal data on eating behaviors and contexts.
npj Digital Medicine. 2025 [date TBD].

BibTeX:

bibtex
Copy
Edit
@article{Shahabi2025SenseWhy,
  author    = {Farzad Shahabi and Bowen Wei and Christopher Romano and Rachel McCloskey and
               Andrew W. Lin and Maryam Pedram and Jennifer Schauer and Taylor Stump and Nabil Alshurafa},
  title     = {Unveiling overeating patterns within digital longitudinal data on eating behaviors and contexts},
  journal   = {npj Digital Medicine},
  year      = {2025},
  note      = {[date TBD]}
}
Cite Key Tools
pyCirclize
pyCirclize (Version 1.10.0) [Computer software]. (2025, August 23). moshi4/pyCirclize. GitHub repository. https://github.com/moshi4/pyCirclize

UMAP
McInnes, L., Healy, J., & Melville, J. (2018).
UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
arXiv preprint arXiv:1802.03426.

XGBoost
Chen, T., & Guestrin, C. (2016).
XGBoost: A scalable tree boosting system.
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

License
This project is released under the MIT License.
You are free to use, modify, and distribute the code with proper attribution. See the LICENSE file for full details.

