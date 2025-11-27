# blend_opt_windows.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(".")
ID_COL = "participant_id"
TARGET_COL = "personality_cluster"

# expected files (feel free to change names if your scripts saved different ones)
FILES = {
    "oof_rf": DATA_DIR / "oof_probas_rf.npy",
    "oof_lgb": DATA_DIR / "oof_probas_lgb.npy",
    "oof_nn": DATA_DIR / "oof_probas_nn.npy",
    "test_rf": DATA_DIR / "test_probas_rf.npy",
    "test_lgb": DATA_DIR / "test_probas_lgb.npy",
    "test_nn": DATA_DIR / "test_probas_nn.npy",
    "train": DATA_DIR / "train.csv",
    "test": DATA_DIR / "test.csv",
    "sample": DATA_DIR / "sample_submission.csv"
}

for k, p in FILES.items():
    if not p.exists() and k not in ("oof_lgb","oof_nn","test_lgb","test_nn"):
        # require RF and train/test for minimum case
        pass

# Load train/test and sample
train = pd.read_csv(FILES["train"])
test_df = pd.read_csv(FILES["test"])
sample = pd.read_csv(FILES["sample"])

y = train[TARGET_COL].values
le = LabelEncoder()
y_int = le.fit_transform(y)
classes = le.classes_
n_classes = len(classes)

# Load OOF/test probs; if some not present, fall back to zeros or skip
def safe_load(p):
    return np.load(p) if p.exists() else None

oof_rf = safe_load(FILES["oof_rf"])
test_rf = safe_load(FILES["test_rf"])

oof_lgb = safe_load(FILES["oof_lgb"])
test_lgb = safe_load(FILES["test_lgb"])

oof_nn = safe_load(FILES["oof_nn"])
test_nn = safe_load(FILES["test_nn"])

# Must have at least RF OOF
if oof_rf is None:
    raise SystemExit("oof_probas_rf.npy is required. Run the RF bagging script to create it.")

# create list of present models
oof_list = []
test_list = []
names = []
if oof_rf is not None:
    oof_list.append(oof_rf); test_list.append(test_rf); names.append("rf")
if oof_lgb is not None:
    oof_list.append(oof_lgb); test_list.append(test_lgb); names.append("lgb")
if oof_nn is not None:
    oof_list.append(oof_nn); test_list.append(test_nn); names.append("nn")

m = len(oof_list)
print("Loaded base models:", names)

# simple coordinate-ascent optimizer for weights (non-negative, sum unconstrained)
def eval_weights(weights):
    # weights: list-like len m
    w = np.array(weights)
    w = np.clip(w, 0.0, 10.0)
    # weighted sum of oofs
    proba = sum(w[i] * oof_list[i] for i in range(m))
    # normalize per-row
    proba = proba / (proba.sum(axis=1, keepdims=True) + 1e-12)
    preds = proba.argmax(axis=1)
    return f1_score(y_int, preds, average="macro")

# initialize
weights = np.array([1.0] * m)
best_score = eval_weights(weights)
print("Start weights:", dict(zip(names, weights)), "score:", best_score)

# coordinate ascent: for each model, grid-search multiplier while keeping others fixed
# do coarse then refine
grids = [np.linspace(0.0, 2.0, 21), np.linspace(0.5, 1.5, 21), np.linspace(0.8, 1.2, 21)]
for grid in grids:
    improved = True
    while improved:
        improved = False
        for i in range(m):
            current = weights[i]
            best_local_val = best_score
            best_local_w = current
            for g in grid:
                trial = weights.copy()
                trial[i] = g
                sc = eval_weights(trial)
                if sc > best_local_val + 1e-12:
                    best_local_val = sc
                    best_local_w = g
            if best_local_w != current:
                weights[i] = best_local_w
                best_score = best_local_val
                improved = True
                print(" updated", names[i], "->", weights[i], "score:", best_score)

print("Final weights:", dict(zip(names, weights)), "best OOF score:", best_score)

# Build final weighted test prob
test_proba = None
for i in range(m):
    tp = test_list[i]
    if tp is None:
        raise SystemExit(f"Missing test prob for {names[i]} (expected file).")
    if test_proba is None:
        test_proba = weights[i] * tp
    else:
        test_proba += weights[i] * tp
test_proba = test_proba / (test_proba.sum(axis=1, keepdims=True) + 1e-12)
test_preds = test_proba.argmax(axis=1)
test_labels = le.inverse_transform(test_preds.astype(int))

# Save submission
out = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: test_labels})
out_path = DATA_DIR / "submission_blend_opt.csv"
out.to_csv(out_path, index=False)
print("Saved submission:", out_path)
