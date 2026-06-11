# Breast Cancer Survival Prediction using METABRIC Dataset

## Overview

This project focuses on the analysis of clinical and molecular data from the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset. The objective was to investigate factors associated with patient outcomes and to build machine learning models predicting the cause of death in breast cancer patients.

## Dataset
Source: https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric

Patients: ~1900 breast cancer cases

Data types:

-Clinical variables
-Mutation profiles
-Survival information

## Workflow
- Exploratory Data Analysis (EDA)
- Missing data analysis and imputation 
- Feature engineering
- Dimensionality reduction using FastICA
- Model development and comparison
- Performance evaluation on an independent test set

## Models Evaluated
- Random Forest
- XGBoost
- CatBoost
- Baseline random classifier
  
Both clinical-only models and combined clinical + mutation models were investigated.

## Main Findings
Combining molecular and clinical information improved predictive performance.
FastICA helped reduce the dimensionality of genomic features.
CatBoost, Random Forest and XGBoost achieved comparable results.
Molecular features provided additional information beyond clinical variables alone.

## Comparison of models using weighted f1-score and accuracy

![](Porownanie%20metryk/Metryki.png)

## Roc curve of best estimated model (Random Forest)

![](Porownanie%20metryk/roc_rf_test.png)

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- Matplotlib
- Seaborn
## Author
- Stola Bartłomiej - preprocessing, modeling and other actions related to data mining and machine learning.
- Lidia Gozdecka - ensuring substantive compliance with medical knowledge
