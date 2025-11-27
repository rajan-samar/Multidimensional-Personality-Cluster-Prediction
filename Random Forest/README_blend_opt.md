
# RF + LGBM + NN Weighted Blend Optimizer (Windows-Friendly)

This README describes the **probability-based ensemble blender** implemented in  
`lgbm_rf_nn_blend.py` (original name: `blend_opt_windows.py`).  fileciteturn10file0  

This script performs **automatic weight optimization** across multiple base model OOF predictions, then applies the optimized weights to test-set probabilities to generate the final blended submission.

It is *different* from stacking or soups — this file focuses on **continuous probability blending + coordinate ascent weight tuning**.

---

## 1. Purpose of This Script

This utility is designed to:

- Combine the predictions of **RandomForest**, **LightGBM**, and **Neural Network** models  
- Use **OOF Macro-F1** as the objective function  
- Automatically search for the best blending weights  
- Normalize blended probabilities row-wise  
- Produce the final submission file:
  ```
  submission_blend_opt.csv
  ```

This forms the **mid-stage ensembling step** in your ML journey — stronger than soups and baselines, but simpler than full meta-learning stacks.

---

## 2. Required Input Files

Place the following files in the same directory:

```
oof_probas_rf.npy     (required)
oof_probas_lgb.npy    (optional)
oof_probas_nn.npy     (optional)

test_probas_rf.npy    (required)
test_probas_lgb.npy   (optional)
test_probas_nn.npy    (optional)

train.csv
test.csv
sample_submission.csv
```

At minimum, RF OOF + RF test probabilities must exist.  
The blend will automatically include models only if both OOF and test probabilities are found.

---

## 3. How Blending Works

### Step 1 — Load available models  
The script checks which models exist:

- RF  
- LGBM  
- NN  

Each model contributes:
- OOF probabilities → used for weight optimization  
- Test probabilities → used for final submission  

### Step 2 — Define the scoring function

```
Macro-F1 of argmax( normalized( sum_i w_i * oof_i ) )
```

Weights are continuous and non-negative.

### Step 3 — Coordinate Ascent Weight Search

For each model:
- Sweep through a series of grids:
  - coarse: 00 → 2.0  
  - mid:    0.5 → 1.5  
  - fine:   0.8 → 1.2  
- Hold other weights fixed  
- Update only if Macro-F1 improves  
- Iterate until no further improvement  

This yields weights like:
```
rf = 1.37
lgb = 0.82
nn = 1.11
```

### Step 4 — Final Test Blending

The final blended probability is:

```
test_proba = Σ_i (weight_i * test_prob_i)
test_proba = normalize(test_proba, axis=1)
pred = argmax(test_proba)
```

### Step 5 — Save Output

```
submission_blend_opt.csv
```

---

## 4. Why This Script Is Useful

This method is:
- **Smarter than simple averaging**
- **Faster and simpler than stacking**
- **Uses OOF performance directly for weight search**
- **Model-agnostic** (works with any `.npy` probability files)
- **High synergy** when RF / LGBM / NN behave differently

It often gives **1–3% Macro-F1 gain** over naïve blends.

---

## 5. Limitations

- Requires at least RF OOF + test probabilities  
- Not a meta-learner — no feature-based stacking  
- Only optimizes linear weights, not nonlinear interactions  

For peak performance, pair this with:
- Meta-learners  
- Stacking pipelines  
- Temperature scaling / calibration  

---

## 6. Final Output Summary

```
submission_blend_opt.csv   ← final ensemble predictions
```

The file contains:
```
participant_id, personality_cluster
```

---

This README is intentionally **focused**, standalone, and specific to the blend optimizer — a different stage in your model journey.

