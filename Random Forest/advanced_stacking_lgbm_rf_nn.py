"""
stacking_advanced_windows.py

Advanced stacking pipeline (Windows-friendly):
 - Bagged RandomForest (OOF probs)
 - LightGBM (tuned, OOF probs)
 - FT-Transformer-like NN (if PyTorch available) or fallback MLP (OOF probs)
 - Meta-model (LightGBM) trained on stacked OOF prob features
 - Optional per-class multiplier greedy refinement
Outputs: submission_stacked_advanced.csv

Place train.csv, test.csv, sample_submission.csv in the same folder.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import time
import warnings
import sys
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

# LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# PyTorch for FT-Transformer-like NN (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# -----------------------
# Config
# -----------------------
DATA_DIR = Path(".")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"
OUT_SUB = DATA_DIR / "submission_stacked_advanced.csv"

ID_COL = "participant_id"
TARGET_COL = "personality_cluster"

RANDOM_STATE = 42
FOLDS = 5
RF_BAGS = 5           # bagged RF count
RF_TUNE = False       # keep tuning off for speed; set True if you want per-bag randomized tuning
N_JOBS = -1

# LightGBM base params (reasonable defaults)
LGB_PARAMS = {
    "objective": "multiclass",
    "metric": "None",
    "num_class": None,  # set later
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "n_estimators": 800,
    "min_data_in_leaf": 20,
    "n_jobs":  -1
}

# Meta-model params (LightGBM)
META_PARAMS = {
    "objective": "multiclass",
    "metric": "None",
    "num_class": None,
    "learning_rate": 0.05,
    "num_leaves": 16,
    "n_estimators": 500,
    "n_jobs": -1
}

# NN training params (if torch available)
NN_EPOCHS = 70
NN_BATCH = 64
NN_LR = 1e-3
DEVICE = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu") if HAS_TORCH else None

# Per-class multiplier coarse grid
CLASS_GRID = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]

# -----------------------
# Sanity checks
# -----------------------
for p in (TRAIN_CSV, TEST_CSV, SAMPLE_CSV):
    if not p.exists():
        raise SystemExit(f"Missing file: {p}. Place train.csv, test.csv, sample_submission.csv in {Path.cwd()} and re-run.")

print("Working directory:", Path.cwd())
print("Torch available:", HAS_TORCH, "LightGBM available:", HAS_LGB)

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_CSV)

if ID_COL not in train.columns or ID_COL not in test.columns or TARGET_COL not in train.columns:
    raise SystemExit("Required columns missing. Ensure participant_id in train/test and personality_cluster in train.")

test_ids = test[ID_COL].values

# -----------------------
# Feature engineering (same FE used previously)
# -----------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"focus_intensity", "consistency_score"}.issubset(df.columns):
        df["focus_consistency_mul"] = df["focus_intensity"] * df["consistency_score"]
        df["focus_consistency_div"] = df["focus_intensity"] / (df["consistency_score"] + 1e-9)
    if {"creative_expression_index", "physical_activity_index"}.issubset(df.columns):
        df["creative_physical_mul"] = df["creative_expression_index"] * df["physical_activity_index"]
    if {"support_environment_score", "external_guidance_usage"}.issubset(df.columns):
        df["support_guidance_ratio"] = df["support_environment_score"] / (df["external_guidance_usage"] + 1e-9)
    if "age_group" in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df["age_group"]):
                df["age_group_bin"] = pd.cut(df["age_group"], bins=[-1,1,2,3,4,10], labels=False)
        except Exception:
            pass
    return df

# -----------------------
# Prepare data
# -----------------------
train_feats = train.drop(columns=[ID_COL]).copy()
y_raw = train_feats.pop(TARGET_COL).values
X = train_feats.copy()
X_test = test.drop(columns=[ID_COL]).copy()

X = feature_engineering(X)
X_test = feature_engineering(X_test)

# numeric-only preprocessing
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", num_cols)
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
from sklearn.compose import ColumnTransformer
preproc = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")
X_proc = preproc.fit_transform(X)
X_test_proc = preproc.transform(X_test)
if hasattr(X_proc, "toarray"): X_proc = X_proc.toarray()
if hasattr(X_test_proc, "toarray"): X_test_proc = X_test_proc.toarray()

le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
n_classes = len(classes)
print("Classes:", list(classes))

# patch LGB params
if HAS_LGB:
    LGB_PARAMS["num_class"] = n_classes
    META_PARAMS["num_class"] = n_classes

n_train = X_proc.shape[0]
n_test = X_test_proc.shape[0]

# Utility to ensure probabilities shape
def ensure_proba_shape(p, n_samples, n_classes):
    arr = np.asarray(p)
    if arr.ndim == 1:
        # predicted labels -> make one-hot
        oh = np.zeros((n_samples, n_classes), dtype=float)
        oh[np.arange(n_samples), arr.astype(int)] = 1.0
        return oh
    if arr.shape != (n_samples, n_classes):
        raise ValueError("Unexpected proba shape", arr.shape, (n_samples, n_classes))
    return arr

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

# -----------------------
# 1) Bagged RandomForest OOF probs (same reliable baseline)
# -----------------------
print("\n[1] Bagged RandomForest OOF generation...")
oof_probas_rf = np.zeros((n_train, n_classes), dtype=float)
test_probas_rf = np.zeros((n_test, n_classes), dtype=float)

for bag in range(RF_BAGS):
    seed = RANDOM_STATE + bag * 17
    oof_b = np.zeros((n_train, n_classes), dtype=float)
    test_b = np.zeros((n_test, n_classes), dtype=float)
    for fold, (tr, va) in enumerate(skf.split(X_proc, y), start=1):
        X_tr, X_va = X_proc[tr], X_proc[va]
        y_tr, y_va = y[tr], y[va]
        rf = RandomForestClassifier(n_estimators=800, min_samples_leaf=2, min_samples_split=5,
                                    max_features="sqrt", class_weight="balanced_subsample",
                                    n_jobs=N_JOBS, random_state=seed + fold)
        rf.fit(X_tr, y_tr)
        proba_va = rf.predict_proba(X_va)
        proba_t = rf.predict_proba(X_test_proc)
        proba_va = ensure_proba_shape(proba_va, len(va), n_classes)
        oof_b[va] = proba_va
        test_b += proba_t / FOLDS
    oof_probas_rf += oof_b
    test_probas_rf += test_b

oof_probas_rf /= RF_BAGS
test_probas_rf /= RF_BAGS
print("RF OOF macro-F1:", f1_score(y, oof_probas_rf.argmax(axis=1), average="macro"))

# -----------------------
# 2) LightGBM base model OOF
# -----------------------
if not HAS_LGB:
    print("\nLightGBM not installed. Install 'lightgbm' for this part. Exiting.")
    raise SystemExit("Install lightgbm (pip install lightgbm) and re-run.")

print("\n[2] LightGBM OOF generation...")
oof_probas_lgb = np.zeros((n_train, n_classes), dtype=float)
test_probas_lgb = np.zeros((n_test, n_classes), dtype=float)

for fold, (tr, va) in enumerate(skf.split(X_proc, y), start=1):
    print(" LGB fold", fold)
    X_tr, X_va = X_proc[tr], X_proc[va]
    y_tr, y_va = y[tr], y[va]

    lgbm = lgb.LGBMClassifier(**LGB_PARAMS, random_state=RANDOM_STATE + fold)

    # FIX: no verbose in fit(), suppress logs with callbacks=[]
    lgbm.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )

    proba_va = lgbm.predict_proba(X_va)
    proba_t = lgbm.predict_proba(X_test_proc)

    oof_probas_lgb[va] = ensure_proba_shape(proba_va, len(va), n_classes)
    test_probas_lgb += proba_t / FOLDS

print("LGB OOF macro-F1:", f1_score(y, oof_probas_lgb.argmax(axis=1), average="macro"))

# -----------------------
# 3) FT-Transformer-like model or fallback MLP OOF
# -----------------------
print("\n[3] Neural base (FT-Transformer-like if torch available, else small MLP)...")
oof_probas_nn = np.zeros((n_train, n_classes), dtype=float)
test_probas_nn = np.zeros((n_test, n_classes), dtype=float)

if HAS_TORCH:
    # Simple tabular Transformer-like encoder: project numeric features -> TransformerEncoder -> pool -> classifier
    class TabTransformer(nn.Module):
        def __init__(self, input_dim, d_model=128, nhead=8, nlayers=2, dim_feedforward=256, dropout=0.1, n_classes=1):
            super().__init__()
            self.fc_in = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, x):
            # x: (batch, features) -> (batch, seq_len=features, 1)? We'll treat features as seq length 1 with d_model
            # Simpler: expand features to tokens by splitting into chunks — but for simplicity, treat as a single token
            # Instead: make tokens = features grouped: represent input as (batch, tokens, d_model) by embedding features dimension
            # We'll map input to (batch, tokens=16, d_model) by linear then reshape
            b, f = x.shape
            h = self.fc_in(x)  # (b, d_model)
            h = h.unsqueeze(1)  # (b,1,d_model)
            h = self.transformer(h)  # (b,1,d_model)
            h = h.mean(dim=1)  # (b, d_model)
            out = self.head(h)
            return out

    def train_nn_get_probas(X_np, y_np, X_test_np, tr_idx, va_idx):
        X_tr = X_np[tr_idx]; y_tr = y_np[tr_idx]
        X_va = X_np[va_idx]; y_va = y_np[va_idx]
        # convert to torch
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
        X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)

        model = TabTransformer(input_dim=X_np.shape[1], d_model=128, nhead=8, nlayers=2, n_classes=n_classes).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=NN_LR, weight_decay=1e-5)

        tr_ds = TensorDataset(X_tr_t, y_tr_t)
        tr_loader = DataLoader(tr_ds, batch_size=NN_BATCH, shuffle=True)
        for epoch in range(NN_EPOCHS):
            model.train()
            for xb, yb in tr_loader:
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            proba_va_raw = torch.softmax(model(X_va_t), dim=1).cpu().numpy()
            proba_test_raw = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
        return proba_va_raw, proba_test_raw

    # run per-fold
    X_all = X_proc.astype(np.float32)
    X_test_all = X_test_proc.astype(np.float32)
    for fold, (tr, va) in enumerate(skf.split(X_all, y), start=1):
        print(" NN fold", fold)
        proba_va, proba_test = train_nn_get_probas(X_all, y, X_test_all, tr, va)
        oof_probas_nn[va] = ensure_proba_shape(proba_va, len(va), n_classes)
        test_probas_nn += proba_test / FOLDS

else:
    # fallback small MLP (sklearn-compatible using simple PyTorch wrapper)
    from sklearn.neural_network import MLPClassifier
    print(" PyTorch not available; using sklearn MLPClassifier fallback (fast).")
    for fold, (tr, va) in enumerate(skf.split(X_proc, y), start=1):
        X_tr, X_va = X_proc[tr], X_proc[va]
        y_tr, y_va = y[tr], y[va]
        mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=600, early_stopping=True, random_state=RANDOM_STATE+fold)
        mlp.fit(X_tr, y_tr)
        proba_va = mlp.predict_proba(X_va)
        proba_test = mlp.predict_proba(X_test_proc)
        oof_probas_nn[va] = ensure_proba_shape(proba_va, len(va), n_classes)
        test_probas_nn += proba_test / FOLDS

print("NN OOF macro-F1:", f1_score(y, oof_probas_nn.argmax(axis=1), average="macro"))

# -----------------------
# Stack base model OOF probas into meta-features
# -----------------------
print("\n[4] Building meta-features and training meta-model...")
# concat probs horizontally: for each base model we provide n_classes probabilities
meta_train = np.hstack([oof_probas_rf, oof_probas_lgb, oof_probas_nn])
meta_test = np.hstack([test_probas_rf, test_probas_lgb, test_probas_nn])
print("Meta train shape:", meta_train.shape, "Meta test shape:", meta_test.shape)

# Meta model: LightGBM if available, else LogisticRegression
if HAS_LGB:
    META_PARAMS["num_class"] = n_classes
    meta = lgb.LGBMClassifier(**META_PARAMS, random_state=RANDOM_STATE)
    meta.fit(meta_train, y)
    meta_oof_preds = meta.predict(meta_train)
    meta_test_proba = meta.predict_proba(meta_test)
else:
    meta = LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced")
    meta.fit(meta_train, y)
    meta_oof_preds = meta.predict(meta_train)
    # get proba via predict_proba
    meta_test_proba = meta.predict_proba(meta_test)

meta_oof_macro = f1_score(y, meta_oof_preds, average="macro")
print("Meta OOF macro-F1 (train-level estimate):", meta_oof_macro)
print("Meta classification report (train-level):")
print(classification_report(le.inverse_transform(y), le.inverse_transform(meta_oof_preds), digits=4))

# -----------------------
# Optional: greedy per-class multiplier tuning on OOF to maximize macro-F1
# -----------------------
print("\n[5] Greedy per-class multiplier tuning on OOF...")
# baseline proba from weighted base combination (we used concat; here we use meta oof probabilities instead)
# For multiplier tuning we will apply multipliers to meta model's predicted probabilities (train-level)
# Build meta probabilities on train via .predict_proba if available; otherwise approximate from `meta_test_proba` - but we have meta trained on meta_train so:
if hasattr(meta, "predict_proba"):
    meta_train_proba = meta.predict_proba(meta_train)
else:
    # fallback: build one-hot from preds
    meta_train_proba = np.zeros((n_train, n_classes), dtype=float)
    meta_train_proba[np.arange(n_train), meta_oof_preds] = 1.0

mults = np.ones(n_classes, dtype=float)
best_score = f1_score(y, meta_train_proba.argmax(axis=1), average="macro")
print(" starting OOF macro-F1:", best_score)
for i in range(n_classes):
    best_local = (1.0, best_score)
    for m in CLASS_GRID:
        trial = mults.copy(); trial[i] = m
        proba = meta_train_proba * trial.reshape(1, -1)
        proba = proba / (proba.sum(axis=1, keepdims=True) + 1e-12)
        sc = f1_score(y, proba.argmax(axis=1), average="macro")
        if sc > best_local[1]:
            best_local = (m, sc)
    mults[i] = best_local[0]
    best_score = best_local[1]
    print(f" class {classes[i]} -> multiplier {mults[i]} new OOF {best_score:.4f}")

print("Final per-class multipliers:", dict(zip(classes, mults.tolist())))
print("OOF after multipliers:", best_score)

# -----------------------
# Final test probabilities: apply meta.predict_proba then multipliers
# -----------------------
if hasattr(meta, "predict_proba"):
    test_meta_proba = meta.predict_proba(meta_test)
else:
    # fallback predict then one-hot
    test_meta_preds = meta.predict(meta_test)
    test_meta_proba = np.zeros((n_test, n_classes), dtype=float)
    test_meta_proba[np.arange(n_test), test_meta_preds] = 1.0

test_meta_proba = test_meta_proba * mults.reshape(1, -1)
test_meta_proba = test_meta_proba / (test_meta_proba.sum(axis=1, keepdims=True) + 1e-12)
test_preds_int = test_meta_proba.argmax(axis=1)
test_preds_labels = le.inverse_transform(test_preds_int)

submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds_labels})
submission.to_csv(OUT_SUB, index=False)
print(f"\nSaved stacked submission: {OUT_SUB}")

# final OOF report (after multipliers)
print("\nFinal OOF report (train-level after multipliers):")
oof_proba_meta = meta_train_proba * mults.reshape(1, -1)
oof_proba_meta = oof_proba_meta / (oof_proba_meta.sum(axis=1, keepdims=True) + 1e-12)
oof_preds_final = oof_proba_meta.argmax(axis=1)
print(classification_report(le.inverse_transform(y), le.inverse_transform(oof_preds_final), digits=4))
print("OOF Macro-F1 (after multipliers):", f1_score(y, oof_preds_final, average="macro"))

print("\nDone.")
