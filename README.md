# SenseWhy
# SenseWhy: Overeating Behavior Analysis

This repository contains the analysis pipeline for the **SenseWhy study**, focusing on predicting, clustering, and visualizing patterns of **overeating behavior** using **sensor features** and **EMA (Ecological Momentary Assessment)** data.

The workflow integrates:
1. **Classification models** (XGBoost, SVM, Naive Bayes, MLP) to predict overeating events.  
2. **Representation learning + clustering** (MLP hidden layers → UMAP → KMeans/DBSCAN) to identify latent overeating clusters.  
3. **Post-hoc analysis (z-score polar plots)** to interpret principal indicators driving overeating clusters.

---

## Repository Structure

```bash
SenseWhy/
│
├── classification/                # Predictive modeling pipeline
│   ├── overeating_sensor_ema_model_eval.ipynb
│   └── overeating_ema_features.csv
│
├── clustering/                    # Representation learning + UMAP + clustering
│   ├── sensewhy_overeating_umap_clusters.ipynb
│   └── res_df_nnout_umap2d_clustered.csv
│
├── posthoc/                       # Z-score analysis & polar plots
│   ├── overeating_clusters_zscore_posthoc_analysis.ipynb
│   └── polarplot_updated_10_21.pdf
│
├── data/                          # Input files (not tracked in repo)
│   ├── overeating_sensor_features.csv
│   ├── overeating_ema_features.csv
│   ├── SenseWhy_demographics.csv
│   ├── SenseWhy_NDSR_Foodtrk_Flagged_220405.csv
│   └── result_table_merged_EMA_experimental_unstyled.xlsx
│
└── README.md                      # This file
