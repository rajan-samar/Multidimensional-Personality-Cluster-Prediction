#!/usr/bin/env python3
"""
Meta-learner tuning and stacking.

Place this script in the same folder that contains directories:
  - oof_probs/    (contains files like mlp_a_oof.csv, mlp_b_oof.csv, ...)
  - test_probs/   (contains corresponding mlp_a_test.csv, ...)

Run:
    python meta_tune_stack.py

Outputs:
  - submission_meta_tuned.csv    (stacked meta prediction using tuned logistic elastic-net)
  - submission_meta_blend.csv    (0.95 stacked + 0.05 oof-weighted ensemble)
  - meta_model.joblib             (saved trained meta model)
  - prints CV grid results and OOF Macro-F1
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from scipy.optimize import minimize
import joblib

ROOT = Path(".").resolve()
OOF_DIR = ROOT / "oof_probs"
TESTPROB_DIR = ROOT / "test_probs"

assert OOF_DIR.exists(), f"{OOF_DIR} does not exist. Run main pipeline first."
assert TESTPROB_DIR.exists(), f"{TESTPROB_DIR} does not exist. Run main pipeline first."

# -------------------------
# Helper: list matching pairs
# -------------------------
oof_files = sorted([p for p in OOF_DIR.glob("*_oof.csv")])
test_files = sorted([p for p in TESTPROB_DIR.glob("*_test.csv")])

# Ensure names match (model keys)
def model_key_from_path(p):
    name = p.name
    # expected format: <model>_oof.csv
    return name.replace("_oof.csv","").replace("_test.csv","").replace(".csv","")

oof_keys = [model_key_from_path(p) for p in oof_files]
test_keys = [model_key_from_path(p) for p in test_files]

common_keys = [k for k in oof_keys if k in test_keys]
if len(common_keys) == 0:
    raise SystemExit("No matching model oof/test pairs found in oof_probs/ and test_probs/.")
print("Found model keys:", common_keys)

# Only keep matching pairs and keep consistent ordering
oof_files = [OOF_DIR / f"{k}_oof.csv" for k in common_keys]
test_files = [TESTPROB_DIR / f"{k}_test.csv" for k in common_keys]

# -------------------------
# Load OOFs -> build meta X_oof, y
# -------------------------
meta_oof_dfs = []
for p in oof_files:
    df = pd.read_csv(p)
    # ensure prob columns present
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if len(prob_cols) == 0:
        raise SystemExit(f"No prob_* columns found in {p}")
    meta_oof_dfs.append(df)

# Validate all have same ordering of participant_id as train ordering? We assume OOF files align on participant_id rowwise.
# We'll use the 'true' column from the first file.
y_true = meta_oof_dfs[0]["true"].values
le = LabelEncoder(); y_enc = le.fit_transform(y_true)
classes = le.classes_.tolist()
n_classes = len(classes)
print("Classes:", classes, "n_classes:", n_classes)

# Build X_oof by concatenating probability columns from each model
X_oof_list = []
for df in meta_oof_dfs:
    pcols = [c for c in df.columns if c.startswith("prob_")]
    X_oof_list.append(df[pcols].values)
# shape: list of (n_samples, n_classes)
# concatenate horizontally
X_oof = np.hstack(X_oof_list)  # shape (n_samples, n_models * n_classes)
print("X_oof shape:", X_oof.shape)

# Save mapping of columns -> model/prob for traceability
model_prob_cols = []
for k, df in zip(common_keys, meta_oof_dfs):
    model_prob_cols.append([f"{k}__{c}" for c in [c for c in df.columns if c.startswith("prob_")]])
flat_columns = [col for sub in model_prob_cols for col in sub]

# -------------------------
# Load test probs -> build X_test
# -------------------------
meta_test_dfs = [pd.read_csv(p) for p in test_files]
# Ensure test prob columns match OOF prob col names 'prob_*' and in same class order
X_test_list = []
for df in meta_test_dfs:
    pcols = [c for c in df.columns if c.startswith("prob_")]
    X_test_list.append(df[pcols].values)
X_test = np.hstack(X_test_list)
print("X_test shape:", X_test.shape)

# Participant ID
test_id_col = meta_test_dfs[0].columns[0]
test_ids = meta_test_dfs[0][test_id_col].values

# -------------------------
# Baseline: compute OOF-weighted ensemble (for blending fallback)
# We'll compute simple average OOF and then find optimal OOF weights (constrained) to maximize OOF macro-F1
# -------------------------
print("\nComputing baseline OOF-weight optimized ensemble (for fallback/blend) ...")
oof_stack_per_model = [df[[c for c in df.columns if c.startswith("prob_")]].values for df in meta_oof_dfs]
n_models = len(oof_stack_per_model)

def loss_weights_obj(w):
    probs = sum(w[i] * oof_stack_per_model[i] for i in range(n_models))
    preds = probs.argmax(axis=1)
    return -f1_score(y_enc, preds, average="macro")

# constraints: weights >=0 sum=1
cons = ({'type':'eq','fun': lambda x: x.sum() - 1.0})
bnds = [(0.0,1.0)]*n_models
x0 = np.ones(n_models) / n_models
res = minimize(loss_weights_obj, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':200})
best_weights = res.x
print("OOF-optimized weights:", dict(zip(common_keys, np.round(best_weights,3).tolist())))
# get weighted test probs for fallback
test_stack_per_model = [df[[c for c in df.columns if c.startswith("prob_")]].values for df in meta_test_dfs]
test_weighted_probs = sum(best_weights[i] * test_stack_per_model[i] for i in range(n_models))
# fallback labels
fallback_labels = le.inverse_transform(test_weighted_probs.argmax(axis=1))

# -------------------------
# Meta hyperparameter grid: logistic elastic-net
# -------------------------
print("\nTuning meta-learner (logistic regression with elastic-net) via GridSearchCV on OOF...")
scorer = make_scorer(f1_score, average='macro')

# Grid: Cs and l1_ratio values tuned conservatively
param_grid = {
    'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    'classifier__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]  # 0 -> L2, 1 -> L1
}
# Build pipeline (we already have probabilities so no scaling)
clf = LogisticRegression(penalty='elasticnet', solver='saga', multi_class='multinomial', max_iter=2000, tol=1e-4)

# We'll use StratifiedKFold for inner CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([('classifier', clf)])
gs = GridSearchCV(pipe, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=1)
gs.fit(X_oof, y_enc)

print("\nGridSearchCV done.")
print("Best params:", gs.best_params_)
print("Best CV (grid) Macro-F1:", gs.best_score_)

# -------------------------
# Train final meta on entire OOF dataset with best params
# -------------------------
print("\nTraining final meta on full OOF with best params...")
best_C = gs.best_params_['classifier__C']
best_l1 = gs.best_params_['classifier__l1_ratio']
final_meta = LogisticRegression(C=best_C, penalty='elasticnet', l1_ratio=best_l1,
                                solver='saga', multi_class='multinomial', max_iter=5000, tol=1e-5)
final_meta.fit(X_oof, y_enc)
# OOF prediction check (sanity)
oof_meta_preds = final_meta.predict(X_oof)
oof_meta_f1 = f1_score(y_enc, oof_meta_preds, average='macro')
print("Final meta OOF Macro-F1 (on OOF training):", oof_meta_f1)

# -------------------------
# Predict test with meta
# -------------------------
test_meta_probs = final_meta.predict_proba(X_test)
test_meta_labels = final_meta.predict(X_test)
test_meta_labels_str = le.inverse_transform(test_meta_labels)

# Save submission
out_df = pd.DataFrame({test_id_col: test_ids, 'personality_cluster': test_meta_labels_str})
out_df.to_csv("submission_meta_tuned.csv", index=False)
print("Saved submission_meta_tuned.csv")

# Save meta model
joblib.dump(final_meta, "meta_model.joblib")

# -------------------------
# Safety blend: 0.95 stacked + 0.05 weighted-OOF fallback (very small hedge)
# -------------------------
print("\nSaving small blend of meta + oof-weighted (0.95/0.05) as safety hedge...")
stack_probs = test_meta_probs
blend_probs = 0.9 * stack_probs + 0.1 * test_weighted_probs
blend_preds = le.inverse_transform(blend_probs.argmax(axis=1))
pd.DataFrame({test_id_col: test_ids, 'personality_cluster': blend_preds}).to_csv("submission_meta_blend.csv", index=False)
print("Saved submission_meta_blend.csv")

# -------------------------
# Diagnostics: per-class confusion on OOF meta
# -------------------------
try:
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nMeta OOF classification report:")
    print(classification_report(y_enc, oof_meta_preds, target_names=le.classes_))
except Exception:
    pass

print("\nAll done. Files created:")
print(" - submission_meta_tuned.csv")
print(" - submission_meta_blend.csv")
print(" - meta_model.joblib")
print("\nIf you want, I will now (1) try a small non-linear meta (LightGBM/XGBoost) OR (2) do per-class thresholding or weighted-F1 meta objective. Which next? (reply with 1 or 2 or 'both' or 'stop')")
