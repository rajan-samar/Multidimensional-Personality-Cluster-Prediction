
# Neural-Network Submission Soup & Meta-Learner Stacking README

This README documents two advanced ensemble utilities:

- **soup_nn.py** — A simple hard-vote *label soup* for classification submissions  
- **nn_meta_tune_stack.py** — A full meta-learning + stacking pipeline with OOF tuning, logistic elastic-net, and blended final predictions  

Both scripts are designed for multi-class ML competitions using pre-generated model outputs.

---

# 1. submission soup — `soup_nn.py`

Source: `soup_nn.py`

This script performs a **simple majority-vote / hard-label soup** using several top-performing submission CSVs.  
It assumes:

- Each submission contains final labels (not probabilities)  
- All files share the same ordering and contain:
  - `participant_id`
  - `personality_cluster`  

### Workflow

1. Load four strongest submission files:
   - `submission_meta_tuned.csv`
   - `submission_meta_blend.csv`
   - `submission_stacked.csv`
   - `submission_superstack_blend.csv`
2. Treat each row's label as one-hot.
3. Average all one-hot matrices.
4. Take `argmax` across classes to get final fused prediction.
5. Save:
   ```
   submission_final_soup.csv
   ```

### Notes
- Extremely robust for multi-class problems.
- Simple soup often improves stability.
- Requires consistent order of participants across files.

---

# 2. Meta-Learner + Stacking — `nn_meta_tune_stack.py`

Source: `nn_meta_tune_stack.py`

This is a **full stacking, tuning, and meta-learning pipeline**.

### Inputs
Directory structure:
```
oof_probs/
    modelA_oof.csv
    modelB_oof.csv
    ...
test_probs/
    modelA_test.csv
    modelB_test.csv
    ...
```

Each `*_oof.csv` & `*_test.csv` must contain:
- a `prob_*` column per class
- a `true` column (only in OOF)

### Pipeline Summary

#### 1. Load & align OOF and test probabilities  
- Checks matching model keys across directories  
- Builds:
  - `X_oof`  → stacked OOF prob matrix  
  - `X_test` → stacked test prob matrix  
  - `y_true` → ground truth labels  

#### 2. Compute an OOF-optimized ensemble  
Uses constrained SLSQP optimization:

```
maximize Macro-F1( Σ w_i * model_i_probs )
subject to: w_i ≥ 0 and Σ w_i = 1
```

This produces:
- `best_weights`
- A fallback probability prediction for test set

#### 3. Tune Meta-Learner  
Meta model: **Logistic Regression (Elastic-Net)**  
Search grid:
- C: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
- l1_ratio: [0, 0.25, 0.5, 0.75, 1]

CV: 5-fold Stratified  
Metric: Macro-F1  

Outputs:
- Best parameters
- Best CV score

#### 4. Train final meta model on full OOF data  
Saves:
- `meta_model.joblib`

#### 5. Generate submissions  
Creates:
1. **`submission_meta_tuned.csv`**  
   - Meta model predictions
2. **`submission_meta_blend.csv`**  
   - Blend:
     ```
     final = 0.9 * meta + 0.1 * OOF-optimized ensemble
     ```

### Diagnostics
- Prints classification report (OOF)
- Prints confusion matrix (if available)

### Notes
- Highly expressive stacking strategy
- Meta-learner explores sparsity (L1) and smoothness (L2)
- OOF tuning ensures meta-model does not degrade performance
- Recommended for competitions with multiple strong models

---

# When to Use Which

| Script | Best Use Case |
|--------|---------------|
| **soup_nn.py** | Fast, stable, simple voting across strong submissions |
| **nn_meta_tune_stack.py** | Maximum performance through OOF stacking + tuned logistic meta-learner |

---

# Outputs Produced

- `submission_final_soup.csv`
- `submission_meta_tuned.csv`
- `submission_meta_blend.csv`
- `meta_model.joblib`

---

# Summary

These two scripts together form a powerful ensemble system:

- **soup_nn.py** → Hard-vote stability  
- **nn_meta_tune_stack.py** → Tuned, optimized stacking for best macro-F1  

Perfect for final-stage model fusion in ML competitions.

