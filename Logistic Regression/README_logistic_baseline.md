
# Logistic Regression Baseline — Multiclass Personality Cluster Classifier

This README documents the **Logistic Regression baseline pipeline** implemented in  
`logistic_regression.py` (also named `lr_baseline.py`).  fileciteturn8file0  

It is a **simple, clean, stable starting model** in your ML journey — not the strongest model, but essential because it establishes:

- a reproducible baseline  
- clean preprocessing  
- OOF probability generation  
- a reliable reference for future ensemble/meta-learning steps  

---

## 1. Purpose of This Script

This baseline is designed to:
- Provide **OOF predictions** for stacking or blending later  
- Generate **test-set probabilities** for ensemble models  
- Offer a **fast and interpretable** multiclass classifier  
- Track early model performance before moving to neural nets, LGBM, XGB, stacks, soups, etc.

It plays a small but meaningful role in the overall journey.

---

## 2. Data & Setup

The script expects three files in the working directory:

```
train.csv
test.csv
sample_submission.csv
```

Key assumptions:
- ID column: `participant_id`
- Target column: `personality_cluster`
- Multiclass classification setup

---

## 3. Feature Engineering (Light, Safe)

The baseline applies compact FE to avoid overfitting:

### ✓ Numeric auto-detection  
Numeric features are detected from the dataset automatically.

### ✓ Interaction Features  
Conservative pairwise interactions among:
- focus_intensity  
- consistency_score  
- support_environment_score  
- creative_expression_index  
- physical_activity_index  

Only small controlled combos used.

### ✓ Power Transforms  
For the same features:
- square  
- square-root  

### ✓ Rank-based Features  
Each numeric column gets a normalized rank feature (useful for LR).

### ✓ Full NA handling  
All missing values are filled safely.

This FE is intentionally *minimal*, because LR can become unstable with overly large feature expansions.

---

## 4. Model Training

### Algorithm
**Multinomial Logistic Regression** with:
- `solver = saga`
- `penalty = l2`
- `C = 1.0`
- `max_iter = 600`
- `n_jobs = -1` (full CPU parallelism)

### Cross-Validation
- **5-Fold Stratified CV**
- OOF (out-of-fold) probabilities collected for entire train set  
- Test-set probabilities averaged across folds  

Outputs:
- OOF Macro-F1 per fold  
- Overall OOF Macro-F1  
- Classification report  

This makes the baseline usable in stacking/meta-learning pipelines.

---

## 5. Output Files

The script generates:

### 1. **OOF probabilities**
```
oof_probas_lr.npy
```

### 2. **Test-set probabilities**
```
test_probas_lr.npy
```

### 3. **Submission file**
```
submission_lr.csv
```

Format:
```
participant_id, personality_cluster
```

These outputs are directly compatible for:
- Soups  
- Weighted ensemble  
- Meta learner input  
- Neural net stacking pipelines  

---

## 6. Why This Baseline Matters

Even though later models are far stronger, this LR script is valuable because:

- It produces **cleanly aligned OOF/test probabilities**  
- It helps validate FE correctness  
- It acts as a stable *reference point*  
- It can improve final stacks slightly through diversity  
- It plays a role in your **model journey** without overshadowing advanced models

---

## 7. Summary

This LR script is a **low-complexity, high-utility** component in your larger ML workflow.  
Use it for:
- Quick checks  
- Early benchmarking  
- Feeding stacking pipelines  
- Building diverse ensemble inputs  

It is not the final model — it is the *starting stone*.

