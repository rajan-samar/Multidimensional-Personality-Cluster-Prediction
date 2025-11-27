#!/usr/bin/env python3
"""
full_stack_pipeline_local.py

Full end-to-end pipeline (local-relative paths).
Place this file in the same folder as train.csv and test.csv and run:
    python full_stack_pipeline_local.py

Outputs (created in the same folder):
- models_full_pipeline/      (checkpoints)
- oof_probs/                 (per-model OOF CSVs)
- test_probs/                (per-model test prob CSVs)
- submission_ensemble_avg.csv
- submission_ooftuned.csv
- submission_stacked.csv
- submission_temp_scaled.csv
- submission_distilled.csv
"""

import os, sys, time, warnings, math, json
warnings.filterwarnings("ignore")
from pathlib import Path

# Use relative paths (current working directory)
ROOT = Path(".").resolve()
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH  = ROOT / "test.csv"

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise SystemExit(f"train.csv or test.csv not found in {ROOT}")

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

# -------------------------
# CONFIG
# -------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

OUT_MODELS = ROOT / "models_full_pipeline"
OOF_DIR = ROOT / "oof_probs"
TESTPROB_DIR = ROOT / "test_probs"
OUT_MODELS.mkdir(exist_ok=True)
OOF_DIR.mkdir(exist_ok=True)
TESTPROB_DIR.mkdir(exist_ok=True)

# Hyperparams
N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
GAUSS_NOISE_STD = 0.01
BAG_SAMPLE_RATIO = 0.85
N_BAGS = 5

FE_INTERACTIONS = [
    ("focus_intensity","consistency_score"),
    ("physical_activity_index","hobby_engagement_level"),
    ("creative_expression_index","altruism_score"),
    ("support_environment_score","external_guidance_usage")
]

ARCHS = {
    "mlp_a": (256,128,64),
    "mlp_b": (128,64),
    "mlp_c": (512,256,128)
}
FT_CONFIG = {"d_model": 64, "n_heads": 4, "n_layers": 1, "dropout": 0.10}
STUDENT_ARCH = (128,64)

# -------------------------
# FE Utilities
# -------------------------
def add_simple_features(df, base_cols):
    df2 = df.copy()
    for a,b in FE_INTERACTIONS:
        if a in df2.columns and b in df2.columns:
            df2[f"{a}_x_{b}"] = df2[a] * df2[b]
    for c in base_cols:
        if c in df2.columns:
            df2[f"{c}_sq"] = df2[c] * df2[c]
            df2[f"sqrt_abs_{c}"] = np.sqrt(np.abs(df2[c].fillna(0)) + 1.0)
            if df2[c].min() > -0.9:
                df2[f"log1p_{c}"] = np.log1p(df2[c].fillna(0))
    return df2

# -------------------------
# Load & prepare data
# -------------------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
TARGET = "personality_cluster"
IDCOL = "participant_id" if "participant_id" in test.columns else test.columns[0]

# base features
drop_cols = []
for c in ["participant_id","record_code"]:
    if c in train.columns: drop_cols.append(c)
base_features = [c for c in train.columns if c not in drop_cols + [TARGET]]

# impute median
imputer = SimpleImputer(strategy="median")
X_base = pd.DataFrame(imputer.fit_transform(train[base_features]), columns=base_features)
X_test_base = pd.DataFrame(imputer.transform(test[base_features]), columns=base_features)

X_fe = add_simple_features(X_base, base_features)
X_test_fe = add_simple_features(X_test_base, base_features)
FEATURES = X_fe.columns.tolist()
print("Num features after FE:", len(FEATURES))

# labels
le = LabelEncoder()
y = le.fit_transform(train[TARGET].values)
classes = le.classes_.tolist()
n_classes = len(classes)
counts = Counter(y)
n = len(y)
class_weights = np.array([n/(n_classes * counts[i]) for i in range(n_classes)], dtype=np.float32)
print("Classes:", classes, "counts:", counts)

# -------------------------
# Model classes (PyTorch)
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, units, dropout0=0.3):
        super().__init__()
        layers = []
        for i,u in enumerate(units):
            inp = in_dim if i==0 else units[i-1]
            layers.append(nn.Linear(inp,u))
            layers.append(nn.BatchNorm1d(u))
            layers.append(nn.ReLU())
            drop = dropout0 if i==0 else max(0.1, dropout0/2)
            layers.append(nn.Dropout(drop))
        layers.append(nn.Linear(units[-1], n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class FTTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_features)])
        self.layers = nn.ModuleList([self._make_layer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2, n_classes))
    def _make_layer(self, d_model, n_heads, dropout):
        att = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        ff = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model*2, d_model))
        return nn.ModuleDict({"att": att, "ff": ff, "ln1": nn.LayerNorm(d_model), "ln2": nn.LayerNorm(d_model)})
    def forward(self, x):
        B, F = x.shape
        tokens = []
        for i in range(F):
            t = x[:, i].unsqueeze(1)
            t = self.projs[i](t)
            t = self.norms[i](t)
            tokens.append(t.unsqueeze(1))
        x_tok = torch.cat(tokens, dim=1)
        for layer in self.layers:
            att_out, _ = layer["att"](x_tok, x_tok, x_tok)
            x_tok = layer["ln1"](x_tok + att_out)
            ff_out = layer["ff"](x_tok)
            x_tok = layer["ln2"](x_tok + ff_out)
        pooled = x_tok.mean(dim=1)
        return self.head(pooled)

# -------------------------
# Training helper
# -------------------------
def fit_torch_model_full(model, Xtr, ytr, Xval, yval, Xtest):
    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(Xval, dtype=torch.float32), torch.tensor(yval, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(Xtest, dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = model.to(DEVICE).float()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_f1, no_imp, best_state = -1.0, 0, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE).float()
            if GAUSS_NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * GAUSS_NOISE_STD
            yb = yb.to(DEVICE).long()
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        preds_val = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE).float()
                preds_val.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        preds_val = np.vstack(preds_val)
        val_f1 = f1_score(yval, preds_val.argmax(axis=1), average="macro")
        if val_f1 > best_f1 + 1e-8:
            best_f1 = val_f1; best_state = model.state_dict(); no_imp = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    # final preds
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE).float()
            val_preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    val_preds = np.vstack(val_preds)
    test_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb[0].to(DEVICE).float()
            test_preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    test_preds = np.vstack(test_preds)
    return best_f1, val_preds, test_preds, model

# -------------------------
# Save prob helpers
# -------------------------
def save_probs_oof(model_name, idxs, probs, true_labels=None):
    cols = [f"prob_{c}" for c in classes]
    df = pd.DataFrame(probs, columns=cols)
    df.insert(0, "participant_id", train.iloc[idxs][IDCOL].values)
    if true_labels is not None:
        df["true"] = [classes[t] for t in true_labels]
    path = OOF_DIR / f"{model_name}_oof.csv"
    df.to_csv(path, index=False)
    return str(path)

def save_probs_test(model_name, probs):
    cols = [f"prob_{c}" for c in classes]
    df = pd.DataFrame(probs, columns=cols)
    df.insert(0, "participant_id", test[IDCOL].values)
    path = TESTPROB_DIR / f"{model_name}_test.csv"
    df.to_csv(path, index=False)
    return str(path)

# -------------------------
# Train MLP ensemble (CV)
# -------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
model_oof_paths = {}
model_test_paths = {}
model_fold_scores = {}

print("\n=== Training MLPs with CV ===")
for name, arch in ARCHS.items():
    print(f"\n-- {name} arch {arch}")
    oof_preds = np.zeros((len(X_fe), n_classes), dtype=np.float32)
    test_preds_acc = np.zeros((len(X_test_fe), n_classes), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), start=1):
        print(f" Fold {fold}/{N_FOLDS}")
        Xtr = X_fe.iloc[tr_idx].values.copy()
        Xval = X_fe.iloc[val_idx].values.copy()
        Xtest = X_test_fe.values.copy()
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xval_s = scaler.transform(Xval); Xtest_s = scaler.transform(Xtest)
        model = MLP(in_dim=Xtr_s.shape[1], units=list(arch))
        best_f1, val_preds, test_preds_fold, trained_model = fit_torch_model_full(model, Xtr_s, y[tr_idx], Xval_s, y[val_idx], Xtest_s)
        print(f"  Fold {fold} best val F1: {best_f1:.4f}")
        oof_preds[val_idx] = val_preds
        test_preds_acc += test_preds_fold
        fold_scores.append(best_f1)
        torch.save(trained_model.state_dict(), str(OUT_MODELS / f"{name}_fold{fold}.pt"))
    test_preds_avg = test_preds_acc / N_FOLDS
    oof_path = save_probs_oof(name, np.arange(len(X_fe)), oof_preds, true_labels=y)
    test_path = save_probs_test(name, test_preds_avg)
    model_oof_paths[name] = oof_path
    model_test_paths[name] = test_path
    model_fold_scores[name] = (np.mean(fold_scores), np.std(fold_scores))
    print(f"Model {name} mean CV: {model_fold_scores[name][0]:.4f} ± {model_fold_scores[name][1]:.4f}")

# -------------------------
# Train small FT-Transformer with CV (keep only if helpful)
# -------------------------
print("\n=== Training FT-Transformer (small) ===")
name = "ft_small"
oof_preds = np.zeros((len(X_fe), n_classes), dtype=np.float32)
test_preds_acc = np.zeros((len(X_test_fe), n_classes), dtype=np.float32)
fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), start=1):
    print(f" Fold {fold}/{N_FOLDS}")
    Xtr = X_fe.iloc[tr_idx].values.copy(); Xval = X_fe.iloc[val_idx].values.copy(); Xtest = X_test_fe.values.copy()
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr); Xval_s = scaler.transform(Xval); Xtest_s = scaler.transform(Xtest)
    model = FTTransformer(num_features=Xtr_s.shape[1], **FT_CONFIG)
    best_f1, val_preds, test_preds_fold, trained_model = fit_torch_model_full(model, Xtr_s, y[tr_idx], Xval_s, y[val_idx], Xtest_s)
    print(f"  Fold {fold} best val F1: {best_f1:.4f}")
    oof_preds[val_idx] = val_preds
    test_preds_acc += test_preds_fold
    fold_scores.append(best_f1)
    torch.save(trained_model.state_dict(), str(OUT_MODELS / f"{name}_fold{fold}.pt"))
test_preds_avg = test_preds_acc / N_FOLDS
ft_oof_f1 = f1_score(y, oof_preds.argmax(axis=1), average="macro")
print(f"FT mean CV F1: {np.mean(fold_scores):.4f} (OOF combined: {ft_oof_f1:.4f})")
ft_oof_path = save_probs_oof(name, np.arange(len(X_fe)), oof_preds, true_labels=y)
ft_test_path = save_probs_test(name, test_preds_avg)

# Decide inclusion
mlp_mean = np.mean([model_fold_scores[m][0] for m in model_fold_scores])
ft_mean = np.mean(fold_scores)
include_ft = False
if ft_mean >= mlp_mean - 0.005:
    include_ft = True
    model_oof_paths[name] = ft_oof_path
    model_test_paths[name] = ft_test_path
    model_fold_scores[name] = (ft_mean, np.std(fold_scores))
    print("Including FT in ensemble.")
else:
    print("FT underperformed vs MLPs; NOT including FT in main ensemble (saved for analysis).")

# -------------------------
# Proper bagging with OOB evaluation (no leakage)
# -------------------------
print("\n=== Proper bagging (OOB-based) ===")
mlp_oofs = [pd.read_csv(model_oof_paths[m]) for m in ARCHS.keys()]
mlp_oof_probs = np.stack([m[[c for c in m.columns if c.startswith("prob_")]].values for m in mlp_oofs], axis=0)
ensemble_mlps_oof = np.mean(mlp_oof_probs, axis=0)
baseline_ensemble_oof_f1 = f1_score(y, ensemble_mlps_oof.argmax(axis=1), average="macro")
print("Baseline MLP ensemble OOF Macro-F1:", baseline_ensemble_oof_f1)

bag_included = []
for b in range(1, N_BAGS+1):
    print(f"\nBag {b}/{N_BAGS}")
    idxs = np.random.choice(np.arange(len(X_fe)), size=int(BAG_SAMPLE_RATIO*len(X_fe)), replace=True)
    oob_mask = np.ones(len(X_fe), dtype=bool)
    oob_mask[idxs] = False
    oob_idxs = np.where(oob_mask)[0]
    if len(oob_idxs) < 50:
        print("  Too few OOB samples; skipping bag")
        continue
    X_tr = X_fe.iloc[idxs].values; y_tr = y[idxs]
    X_oob = X_fe.iloc[oob_idxs].values; y_oob = y[oob_idxs]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr); X_oob_s = scaler.transform(X_oob); X_test_s = scaler.transform(X_test_fe.values)
    model = MLP(in_dim=X_tr_s.shape[1], units=list(list(ARCHS.values())[0]))
    best_f1, oob_preds, test_preds, trained_model = fit_torch_model_full(model, X_tr_s, y_tr, X_oob_s, y_oob, X_test_s)
    oob_f1 = f1_score(y_oob, oob_preds.argmax(axis=1), average="macro")
    print(f"  Bag {b} OOB F1: {oob_f1:.4f}")
    if oob_f1 >= baseline_ensemble_oof_f1 - 0.01:
        path_test = save_probs_test(f"bag{b}", test_preds)
        model_test_paths[f"bag{b}"] = path_test
        bag_included.append((b, oob_f1))
        torch.save(trained_model.state_dict(), str(OUT_MODELS / f"bag{b}.pt"))
    else:
        print("  Bag failed OOB threshold; skipping inclusion.")
print("Included bags:", bag_included)

# -------------------------
# Build ensemble list
# -------------------------
models_to_use = list(ARCHS.keys())
if include_ft: models_to_use.append("ft_small")
models_to_use += [k for k in model_test_paths.keys() if k.startswith("bag")]
print("Models to ensemble:", models_to_use)

# load per-model test & oof probs
test_probs_map = {}
oof_probs_map = {}
for m in models_to_use:
    tp = pd.read_csv(TESTPROB_DIR / f"{m}_test.csv")
    tcols = [c for c in tp.columns if c.startswith("prob_")]
    test_probs_map[m] = tp[tcols].values
    oof_file = OOF_DIR / f"{m}_oof.csv"
    if oof_file.exists():
        op = pd.read_csv(oof_file)
        oof_probs_map[m] = op[[c for c in op.columns if c.startswith("prob_")]].values

# simple average ensemble
stack_test = np.stack([test_probs_map[m] for m in models_to_use], axis=0)
ensemble_test_avg = np.mean(stack_test, axis=0)
pred_labels = le.inverse_transform(ensemble_test_avg.argmax(axis=1))
pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred_labels}).to_csv(ROOT / "submission_ensemble_avg.csv", index=False)
print("Saved submission_ensemble_avg.csv")

available_oofs = [m for m in models_to_use if m in oof_probs_map]
stack_oof = np.stack([oof_probs_map[m] for m in available_oofs], axis=0)
ensemble_oof_avg = np.mean(stack_oof, axis=0)
ensem_oof_f1 = f1_score(y, ensemble_oof_avg.argmax(axis=1), average="macro")
print("Ensemble average OOF Macro-F1 (available models):", ensem_oof_f1)

# -------------------------
# OOF-weight optimization
# -------------------------
print("\n=== OOF-weight optimization ===")
oof_stack_list = [oof_probs_map[m] for m in available_oofs]
n_models_eff = len(oof_stack_list)
def loss_weights(w):
    w = np.array(w)
    probs = sum(w[i]*oof_stack_list[i] for i in range(n_models_eff))
    preds = probs.argmax(axis=1)
    return -f1_score(y, preds, average="macro")
cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1.0})
bnds = [(0.0,1.0)]*n_models_eff
x0 = np.ones(n_models_eff)/n_models_eff
res = minimize(loss_weights, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':200})
best_w = res.x
print("Best OOF weights:", dict(zip(available_oofs, best_w.round(3).tolist())))
test_stack_list = [test_probs_map[m] for m in available_oofs]
ensemble_test_weighted = sum(best_w[i]*test_stack_list[i] for i in range(n_models_eff))
pred_labels_w = le.inverse_transform(ensemble_test_weighted.argmax(axis=1))
pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred_labels_w}).to_csv(ROOT / "submission_ooftuned.csv", index=False)
print("Saved submission_ooftuned.csv")

# -------------------------
# Stacking (logistic meta)
# -------------------------
print("\n=== Stacking (logistic meta) ===")
oof_meta = np.hstack([oof_probs_map[m] for m in available_oofs])
meta_clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='saga')
meta_clf.fit(oof_meta, y)
test_meta = np.hstack([test_probs_map[m] for m in available_oofs])
meta_preds = meta_clf.predict(test_meta)
pred_labels_meta = le.inverse_transform(meta_preds)
pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred_labels_meta}).to_csv(ROOT / "submission_stacked.csv", index=False)
meta_oof_preds = meta_clf.predict(oof_meta)
meta_oof_f1 = f1_score(y, meta_oof_preds, average="macro")
print("Stacker OOF Macro-F1:", meta_oof_f1)
print("Saved submission_stacked.csv")

# -------------------------
# Temperature scaling (grid search on OOF)
# -------------------------
print("\n=== Temperature scaling ===")
def apply_temp(probs, T):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = np.exp(logits / T)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    return scaled

bestT, bestF = 1.0, ensem_oof_f1
for T in np.linspace(0.5, 2.0, 16):
    scaled = apply_temp(ensemble_oof_avg, T)
    f = f1_score(y, scaled.argmax(axis=1), average="macro")
    if f > bestF:
        bestF, bestT = f, T
print("Best temp:", bestT, "best OOF F1 after temp:", bestF)
ensemble_test_temp = apply_temp(ensemble_test_weighted, bestT)
pred_labels_temp = le.inverse_transform(ensemble_test_temp.argmax(axis=1))
pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred_labels_temp}).to_csv(ROOT / "submission_temp_scaled.csv", index=False)
print("Saved submission_temp_scaled.csv")

# -------------------------
# Distillation (teacher = weighted ensemble)
# -------------------------
print("\n=== Distillation ===")
teacher_oof_probs = sum(best_w[i]*oof_stack_list[i] for i in range(n_models_eff))
# Train student
def train_student(X, y_true, teacher_probs, X_test):
    X_train, X_val, y_tr, y_val, t_tr, t_val = train_test_split(X, y_true, teacher_probs, test_size=0.2, random_state=SEED, stratify=y_true)
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train); Xval_s = scaler.transform(X_val); Xtest_s = scaler.transform(X_test)
    student = MLP(in_dim=Xtr_s.shape[1], units=list(STUDENT_ARCH)).to(DEVICE).float()
    opt = AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    alpha = 0.5; T = 2.0
    tr_ds = TensorDataset(torch.tensor(Xtr_s,dtype=torch.float32), torch.tensor(y_tr,dtype=torch.long), torch.tensor(t_tr,dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(Xval_s,dtype=torch.float32), torch.tensor(y_val,dtype=torch.long), torch.tensor(t_val,dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True); va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)
    for epoch in range(1, 201):
        student.train()
        for xb, yb, t_soft in tr_loader:
            xb = xb.to(DEVICE).float(); yb = yb.to(DEVICE).long(); t_soft = t_soft.to(DEVICE).float()
            logits = student(xb)
            loss_hard = ce(logits, yb)
            s_log = torch.log_softmax(logits / T, dim=1)
            t_soft_norm = (t_soft / (t_soft.sum(dim=1, keepdim=True) + 1e-12)).to(DEVICE)
            loss_soft = torch.mean(torch.sum(- t_soft_norm * s_log, dim=1))
            loss = alpha * loss_hard + (1.0 - alpha) * (T*T) * loss_soft
            opt.zero_grad(); loss.backward(); opt.step()
    # predict test
    test_ds = DataLoader(TensorDataset(torch.tensor(X_test,dtype=torch.float32)), batch_size=BATCH_SIZE)
    preds = []
    student.eval()
    with torch.no_grad():
        for xb in test_ds:
            xb = xb[0].to(DEVICE).float()
            preds.append(torch.softmax(student(xb), dim=1).cpu().numpy())
    preds = np.vstack(preds)
    return preds

scaler_all = StandardScaler().fit(X_fe.values)
X_all_s = scaler_all.transform(X_fe.values)
X_test_s = scaler_all.transform(X_test_fe.values)
student_test_preds = train_student(X_all_s, y, teacher_oof_probs, X_test_s)
pd.DataFrame({IDCOL: test[IDCOL], TARGET: le.inverse_transform(student_test_preds.argmax(axis=1))}).to_csv(ROOT / "submission_distilled.csv", index=False)
print("Saved submission_distilled.csv")

print("\nAll done. Outputs saved to current folder:")
print("  - models_full_pipeline/")
print("  - oof_probs/")
print("  - test_probs/")
print("  - submission_*.csv")
