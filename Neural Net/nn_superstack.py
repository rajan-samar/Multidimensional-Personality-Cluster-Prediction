#!/usr/bin/env python3
"""
meta_lgbm_superstack.py

Full nonlinear meta-stack using LightGBM + engineered meta-features.
Works on ALL LightGBM versions (uses callback-style early stopping).

Required directories:
  oof_probs/
  test_probs/
Generated files:
  submission_superstack_lgbm.csv
  submission_superstack_sharpened.csv
  submission_superstack_blend.csv
  meta_lgbm_model.joblib
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

ROOT = Path(".").resolve()
OOF_DIR = ROOT / "oof_probs"
TEST_DIR = ROOT / "test_probs"

if not OOF_DIR.exists() or not TEST_DIR.exists():
    raise SystemExit("ERROR: Missing oof_probs/ or test_probs/ directory.")

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def load_pairs():
    oof_files = sorted(OOF_DIR.glob("*_oof.csv"))
    test_files = sorted(TEST_DIR.glob("*_test.csv"))
    def key(p): return p.name.replace("_oof.csv","").replace("_test.csv","")
    oof_map = {key(f): f for f in oof_files}
    test_map = {key(f): f for f in test_files}
    keys = sorted([k for k in oof_map if k in test_map])
    return keys, [oof_map[k] for k in keys], [test_map[k] for k in keys]

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1)
    return -np.sum(p*np.log(p), axis=1)

def top2_gap(p):
    s = np.sort(p, axis=1)[:, ::-1]
    return s[:,0] - s[:,1]

def consensus_label(preds):  # preds: (n_models,n_samples)
    import scipy.stats as ss
    mode, count = ss.mode(preds, axis=0, keepdims=False)
    return (count / preds.shape[0]).ravel()

# --------------------------------------------------------
# Load OOF + TEST probs
# --------------------------------------------------------
keys, oof_paths, test_paths = load_pairs()
print("Model keys:", keys)

oof_dfs = [pd.read_csv(p) for p in oof_paths]
test_dfs = [pd.read_csv(p) for p in test_paths]

y_true = oof_dfs[0]["true"].values
le = LabelEncoder()
y = le.fit_transform(y_true)
classes = le.classes_.tolist()
n_classes = len(classes)

oof_list = [df[[c for c in df.columns if c.startswith("prob_")]].values for df in oof_dfs]
test_list = [df[[c for c in df.columns if c.startswith("prob_")]].values for df in test_dfs]

X_oof_base = np.hstack(oof_list)
X_test_base = np.hstack(test_list)

# --------------------------------------------------------
# Build advanced meta features
# --------------------------------------------------------
n_models = len(keys)
oof_stack = np.stack(oof_list, axis=0)   # (n_models,n_samples,n_classes)
test_stack = np.stack(test_list, axis=0)

# per-model metrics
oof_entropy_m = np.array([entropy(oof_stack[i]) for i in range(n_models)])
test_entropy_m = np.array([entropy(test_stack[i]) for i in range(n_models)])

oof_maxprob_m = np.array([oof_stack[i].max(axis=1) for i in range(n_models)])
test_maxprob_m = np.array([test_stack[i].max(axis=1) for i in range(n_models)])

oof_preds_idx = np.array([oof_stack[i].argmax(axis=1) for i in range(n_models)])
test_preds_idx = np.array([test_stack[i].argmax(axis=1) for i in range(n_models)])

# aggregate prob stats
oof_mean_p = oof_stack.mean(axis=0)
oof_std_p  = oof_stack.std(axis=0)

test_mean_p = test_stack.mean(axis=0)
test_std_p  = test_stack.std(axis=0)

# entropy stats
oof_mean_ent = oof_entropy_m.mean(axis=0)
oof_std_ent  = oof_entropy_m.std(axis=0)

test_mean_ent = test_entropy_m.mean(axis=0)
test_std_ent  = test_entropy_m.std(axis=0)

# maxprob stats
oof_mean_max = oof_maxprob_m.mean(axis=0)
oof_std_max  = oof_maxprob_m.std(axis=0)

test_mean_max = test_maxprob_m.mean(axis=0)
test_std_max  = test_maxprob_m.std(axis=0)

# consensus
oof_cons = consensus_label(oof_preds_idx)
test_cons = consensus_label(test_preds_idx)

# class vote fractions
oof_vote = np.zeros_like(oof_mean_p)
test_vote = np.zeros_like(test_mean_p)
for c in range(n_classes):
    oof_vote[:,c] = (oof_preds_idx == c).sum(axis=0)/n_models
    test_vote[:,c] = (test_preds_idx == c).sum(axis=0)/n_models

# assemble meta
meta_oof = np.hstack([
    X_oof_base,
    oof_mean_p, oof_std_p, oof_vote,
    oof_mean_ent.reshape(-1,1),
    oof_std_ent.reshape(-1,1),
    oof_mean_max.reshape(-1,1),
    oof_std_max.reshape(-1,1),
    oof_cons.reshape(-1,1)
])
meta_test = np.hstack([
    X_test_base,
    test_mean_p, test_std_p, test_vote,
    test_mean_ent.reshape(-1,1),
    test_std_ent.reshape(-1,1),
    test_mean_max.reshape(-1,1),
    test_std_max.reshape(-1,1),
    test_cons.reshape(-1,1)
])

print("X_oof_meta:", meta_oof.shape, "X_test_meta:", meta_test.shape)

# --------------------------------------------------------
# Quick Param Search (compatible with old LightGBM)
# --------------------------------------------------------
param_grid = {
    'num_leaves': [31, 63],
    'learning_rate': [0.02, 0.05],
    'min_child_samples': [10,20],
    'reg_alpha': [0.0,0.2],
    'reg_lambda': [0.0,0.2]
}

def quick_search(X, y, grid, seed):
    print("\nRunning quick param search...")
    best_s = -1
    best_p = None
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for nl in grid['num_leaves']:
        for lr in grid['learning_rate']:
            for mcs in grid['min_child_samples']:
                for ra in grid['reg_alpha']:
                    for rl in grid['reg_lambda']:
                        params = {
                            'objective': 'multiclass',
                            'num_class': n_classes,
                            'num_leaves': nl,
                            'learning_rate': lr,
                            'min_child_samples': mcs,
                            'reg_alpha': ra,
                            'reg_lambda': rl,
                            'verbose': -1,
                            'seed': seed,
                        }
                        scores=[]
                        for tr,va in skf3.split(X,y):
                            train_ds = lgb.Dataset(X[tr],label=y[tr])
                            val_ds = lgb.Dataset(X[va],label=y[va])
                            bst = lgb.train(
                                params,
                                train_ds,
                                num_boost_round=200,
                                valid_sets=[val_ds],
                                callbacks=[lgb.early_stopping(25), lgb.log_evaluation(False)]
                            )
                            pred = bst.predict(X[va])
                            scores.append(f1_score(y[va], pred.argmax(1), average='macro'))
                        m=np.mean(scores)
                        if m>best_s:
                            best_s=m; best_p=params
    print("Best quick-search score:", best_s)
    return best_p

SEEDS=[42,2023,7]
base_params = quick_search(meta_oof, y, param_grid, seed=42)
print("Selected params:", base_params)

# --------------------------------------------------------
# 3-seed LightGBM meta (OOF)
# --------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_oof=[]
all_test=[]

for sd in SEEDS:
    print(f"\nTraining LightGBM meta seed={sd}")
    params=base_params.copy()
    params['seed']=sd
    oof_pred=np.zeros((meta_oof.shape[0], n_classes))
    test_pred=np.zeros((meta_test.shape[0], n_classes))
    fold_scores=[]
    for f,(tr,va) in enumerate(skf.split(meta_oof,y),1):
        tr_ds=lgb.Dataset(meta_oof[tr],label=y[tr])
        va_ds=lgb.Dataset(meta_oof[va],label=y[va])
        bst=lgb.train(
            params,
            tr_ds,
            num_boost_round=3000,
            valid_sets=[va_ds],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)]
        )
        pred_va=bst.predict(meta_oof[va])
        oof_pred[va]=pred_va
        fold_scores.append(f1_score(y[va], pred_va.argmax(1), average='macro'))
        test_pred+=bst.predict(meta_test)
    test_pred/=5
    print(f" Seed {sd} CV mean F1:{np.mean(fold_scores):.4f}")
    all_oof.append(oof_pred)
    all_test.append(test_pred)

# average seeds
oof_ens = np.mean(all_oof,axis=0)
test_ens= np.mean(all_test,axis=0)

print("\n=== Final OOF Macro-F1:", f1_score(y, oof_ens.argmax(1), average='macro'))
print(classification_report(y, oof_ens.argmax(1), target_names=classes))

# --------------------------------------------------------
# Save raw superstack
# --------------------------------------------------------
ids = test_dfs[0].iloc[:,0].values
raw_labels = le.inverse_transform(test_ens.argmax(1))
pd.DataFrame({"participant_id": ids, "personality_cluster": raw_labels}).to_csv("submission_superstack_lgbm.csv",index=False)
print("Saved submission_superstack_lgbm.csv")

# --------------------------------------------------------
# Per-class sharpening
# --------------------------------------------------------
def score_m(m):
    m=np.clip(m,0.5,2)
    scaled=oof_ens*m.reshape(1,-1)
    scaled/=scaled.sum(1,keepdims=True)
    return -f1_score(y, scaled.argmax(1), average='macro')

print("\nOptimizing class multipliers...")
res=minimize(score_m, np.ones(n_classes), method='Nelder-Mead', options={'maxiter':200})
best_m=np.clip(res.x,0.5,2)
print("Best multipliers:", np.round(best_m,3))

scaled_test=test_ens*best_m.reshape(1,-1)
scaled_test/=scaled_test.sum(1,keepdims=True)
sharp_labels=le.inverse_transform(scaled_test.argmax(1))

pd.DataFrame({"participant_id": ids, "personality_cluster": sharp_labels}).to_csv("submission_superstack_sharpened.csv", index=False)
print("Saved submission_superstack_sharpened.csv")

# --------------------------------------------------------
# Optional blend (if previous meta_tuned exists)
# --------------------------------------------------------
blend_path=None
for p in ["submission_meta_tuned.csv","submission_ensemble_avg.csv"]:
    if Path(p).exists():
        blend_path=p
        break

if blend_path:
    print("\nBlend reference found:", blend_path)
    ref=pd.read_csv(blend_path).set_index("participant_id").loc[ids]
    oh=np.zeros_like(test_ens)
    idx=[classes.index(x) for x in ref["personality_cluster"].values]
    oh[np.arange(len(idx)),idx]=1
    blended=0.95*test_ens + 0.05*oh
    blend_labels=le.inverse_transform(blended.argmax(1))
    pd.DataFrame({"participant_id":ids,"personality_cluster":blend_labels}).to_csv("submission_superstack_blend.csv",index=False)
    print("Saved submission_superstack_blend.csv")

# --------------------------------------------------------
# Save final meta model on full data
# --------------------------------------------------------
final_ds=lgb.Dataset(meta_oof,label=y)
final_model=lgb.train(base_params, final_ds, num_boost_round=300)
joblib.dump(final_model,"meta_lgbm_model.joblib")
print("Saved meta_lgbm_model.joblib")

print("\nDONE!")
print("Generated:")
print(" - submission_superstack_lgbm.csv")
print(" - submission_superstack_sharpened.csv")
if blend_path: print(" - submission_superstack_blend.csv")
print(" - meta_lgbm_model.joblib")
