# Multidimensional-Personality-Cluster-Prediction

# Overview
This project predicts participant personality clusters based on anonymized behavioral and lifestyle features.

## Project Structure
```
ML_Personality_Cluster_Project/
│
├── models/
│   ├── mlp_a.py
│   ├── mlp_b.py
│   ├── mlp_c.py
│   ├── ft_small.py
│   └── meta_lgbm_model.joblib 
│
├── scripts/
│   ├── train_all_nns.py              # trains NN base models
│   ├── generate_oof_test_probs.py    # saves OOF + test preds
│   ├── meta_stack_logreg.py
│   ├── meta_superstack_lgbm.py
│   ├── make_final_soup.py            # FINAL SOUP
│
├── submissions/
│   ├── submission_meta_tuned.csv
│   ├── submission_meta_blend.csv
│   ├── submission_stacked.csv
│   ├── submission_superstack_blend.csv
│   └── submission_final_soup.csv     # BEST (score: 0.639)
│
├── ML_Project_Final_Report.pdf       
└── README.md

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




