# Multidimensional-Personality-Cluster-Prediction

# Overview
This project predicts participant personality clusters based on anonymized behavioral and lifestyle features.

## Project Structure
```
Multidimensional-Personality-Cluster-Prediction/
├── README.md                      # Project overview, quick start, links to scripts
├── data/                      # Original uploaded CSVs 
│   │   ├── train.csv
│   │   └── test.csv
│   └── sample_submission.csv
├── Light_Gradient_Boosting/       # LGBM-specific scripts & outputs
│   ├── train_lgbm_fe_kfold.py
│   ├── tune_lgbm.py
│   └── README.md                  # short description for this folder
├── Logistic_Regression/
│   ├── logistic_regression.py
│   └── README.md
├── Random_Forest/
│   ├── rf_bagging.py
│   └── README.md
├── Neural_Net/
│   ├── neuralnet_fe_enc.py
│   ├── nn_all.py
│   ├── full_stack_pipeline_local.py
│   ├── nn_meta_tune_stack.py
├── Support_Vector_Machine/
│   ├── svm.py
│   └── README.md
├── best_submission/
│   ├── nn_meta_tune_stack.py

```

## Final Leaderboard Score
**0.639 Macro-F1** (Public LB)

## Project Pipeline Overview
1. **Data Preprocessing**
   - Imputation
   - Robust scaling
   - Feature engineering (12 → 52 numeric features)

2. **Neural Network Models**
   - MLP-A (256-128-64)
   - MLP-B (512-256-128)
   - MLP-C (128-64)
   - FT-small transformer

3. **Out-of-Fold Predictions**
   - 5-fold CV
   - OOF probabilities saved for stacking

4. **Ensembling**
   - Weighted average ensemble
   - Logistic regression stacking
   - LightGBM meta-superstack (entropy + consensus features)
   - **4-model elite soup → FINAL SCORE 0.639**

## Approach
- Explored and prepared data using Python files within each model’s folder.
- Implemented a variety of neural net models.
- Evaluated each model’s performance and submitted predictions to Kaggle to track leaderboard scores.
- Documented findings and parameter tuning in each model’s README file.

## Usage
1. Clone the repo.
2. Download Kaggle dataset files (`train.csv`, `test.csv`) and place them in the `data/` folder.
3. Review notebooks in each model folder to understand training and inference details.
4. Run notebooks to reproduce results.

## Evaluation Metric
- F1 Score




