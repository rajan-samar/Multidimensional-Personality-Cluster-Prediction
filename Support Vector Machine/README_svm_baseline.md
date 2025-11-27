
# SVM Baseline — Multiclass Personality Cluster Classifier

This README documents the **SVM baseline training pipeline** implemented in  
`svm.py` (also named `svm_baseline.py`).    

This baseline plays a **different**, independent role in the journey —  
a stable, margin‑based nonlinear model that provides unique probability behaviour, making it valuable for later blending and stacking even if it is not the strongest standalone model.

---

## 1. Purpose of This Script

The SVM baseline is designed to:

- Generate **OOF probabilities** for stacking and meta-learners  
- Produce **test-set probabilities** compatible with soup/stack pipelines  
- Act as a **nonlinear baseline** complementing LR, NN, LGBM, XGB  
- Provide a **diversity-rich classifier** ideal for ensemble gains  

Unlike linear LR or deep NNs, the RBF kernel captures smooth nonlinear boundaries with strong generalisation.

---

## 2. Feature Engineering (Lightweight, Controlled)

The script performs safe and minimal FE:

### ✓ Auto-detected numeric columns  
The model uses all numeric columns except:
- `participant_id`
- `personality_cluster`
- `record_code`

### ✓ Behavioral interaction features  
If present:
- focus_intensity  
- consistency_score  
- support_environment_score  
- creative_expression_index  
- physical_activity_index  

A limited subset of pairwise interactions is created:
- multiplication  
- division  

### ✓ Power transforms  
For selected features:
- square  
- square‑root  

### ✓ Rank-normalized transforms  
Each numeric column gets a rank feature:
```
rank = rankdata(col) / (N+1)
```

### ✓ Missing values  
All NaNs are filled safely with 0.0 after transformations.

These steps remain intentionally small because SVMs can overfit with large FE expansions.

---

## 3. Data Preprocessing

### Feature scaling  
SVM requires scaled inputs — the script applies:

```
StandardScaler()
```

Fit on train, applied to test.

### Label encoding  
`personality_cluster` → integers using LabelEncoder.

---

## 4. Model Training

A **Radial Basis Function (RBF)** SVM is trained:

```
SVC(
    kernel="rbf",
    C=2.0,
    gamma="scale",
    probability=True,
)
```

Key properties:
- **probability=True** enables softmax-like outputs  
- Good for ensembling and stacking  
- `C=2.0` improves boundary flexibility  
- Stratified 5-fold CV used  

For each fold:
- Train on fold-train  
- Predict probabilities for fold-val → stored as OOF  
- Predict probabilities for test → averaged  

---

## 5. Evaluation

After CV:

- OOF Macro‑F1  
- Per-fold Macro‑F1  
- Full classification report  

These metrics help judge standalone performance & meta-learning value.

---

## 6. Output Files

The script saves:

### 1. OOF probabilities
```
oof_probas_svm.npy
```

### 2. Test probabilities
```
test_probas_svm.npy
```

### 3. Final submission CSV
```
submission_svm.csv
```

Format:
```
participant_id, personality_cluster
```

These files integrate directly into:
- soups  
- weighted probability blends  
- stacking meta-learners  
- tuned ensembles  

---

## 7. Why This Baseline Is Useful

Even if stronger models exist later, SVM adds:

### ✔ Nonlinear but smooth probability behavior  
### ✔ High model diversity (essential for stacking)  
### ✔ Stable OOF predictions  
### ✔ Complements tree-based and neural models  

This diversity boosts performance when combining models in:
- super-stackers  
- meta-learners  
- final weighted blends  

---

## 8. Summary

The SVM baseline is a **clean, powerful, nonlinear classifier** that serves as a stable component in your overall ML workflow.  
It stands apart from LR, NN, LGB, and XGB — making your ensemble stronger through probability diversity.

