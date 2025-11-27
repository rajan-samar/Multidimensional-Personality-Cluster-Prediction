#!/usr/bin/env python3
"""
train_nn_ensemble5.py

Ensemble of 5 diverse MLPs (each trained with Stratified 5-fold CV).
Reads:
    /mnt/data/train.csv
    /mnt/data/test.csv
Writes:
    /mnt/data/submission_ensemble5.csv
    /mnt/data/models_ensemble/... (per-model, per-fold)
Prints per-model CV Macro-F1 and overall ensemble OOF Macro-F1.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

# ========== Paths ==========
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUTPUT_SUB = "submission_nn.csv"
MODELS_DIR = "models_nn"
os.makedirs(MODELS_DIR, exist_ok=True)

# ========== Simple FE config ==========
FE_INTERACTIONS = [
    ("focus_intensity", "consistency_score"),
    ("physical_activity_index", "hobby_engagement_level"),
    ("creative_expression_index", "altruism_score"),
    ("support_environment_score", "external_guidance_usage"),
]

def add_simple_features(df, base_cols):
    df2 = df.copy()
    # interactions
    for a, b in FE_INTERACTIONS:
        if a in df2.columns and b in df2.columns:
            df2[f"{a}_x_{b}"] = df2[a] * df2[b]
    # squares
    for c in base_cols:
        if c in df2.columns:
            df2[f"{c}_sq"] = df2[c] * df2[c]
    # sqrt-abs
    for c in base_cols:
        if c in df2.columns:
            df2[f"sqrt_abs_{c}"] = np.sqrt(np.abs(df2[c].fillna(0)) + 1.0)
    # safe log1p
    for c in base_cols:
        if c in df2.columns:
            if df2[c].min() > -0.9:
                df2[f"log1p_{c}"] = np.log1p(df2[c].fillna(0))
    return df2

# ========== Load data ==========
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGET_COL = "personality_cluster"
ID_COL = "participant_id" if "participant_id" in test.columns else test.columns[0]

if TARGET_COL not in train.columns:
    raise SystemExit(f"Target '{TARGET_COL}' not found in {TRAIN_PATH}")

# base numeric features (drop ids if present)
drop_cols = []
for c in ["participant_id", "record_code"]:
    if c in train.columns:
        drop_cols.append(c)
base_features = [c for c in train.columns if c not in drop_cols + [TARGET_COL]]
print("Base features:", base_features)

# Impute base features with median (global)
imputer = SimpleImputer(strategy="median")
X_base = pd.DataFrame(imputer.fit_transform(train[base_features]), columns=base_features)
X_test_base = pd.DataFrame(imputer.transform(test[base_features]), columns=base_features)

# Add FE
X_fe = add_simple_features(X_base, base_features)
X_test_fe = add_simple_features(X_test_base, base_features)
print("Features after FE:", X_fe.shape[1])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(train[TARGET_COL].values)
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))
print("Class distribution:", Counter(y))

# Compute class weights (for PyTorch)
counts = Counter(y)
n = len(y)
class_weights = np.array([n / (num_classes * counts[i]) for i in range(num_classes)], dtype=np.float32)

# ================= Try PyTorch (preferred) else sklearn =================
use_torch = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    use_torch = False
    device = "cpu"

print("Using PyTorch:", use_torch, "| Device:", device)

# ================= Define PyTorch MLP class =================
if use_torch:
    class MLP(nn.Module):
        def __init__(self, in_dim, units):
            super().__init__()
            layers = []
            for i in range(len(units)):
                in_f = in_dim if i == 0 else units[i-1]
                out_f = units[i]
                layers.append(nn.Linear(in_f, out_f))
                layers.append(nn.BatchNorm1d(out_f))
                layers.append(nn.ReLU())
                # dropout schedule: heavier early, lighter later
                drop = 0.35 if i == 0 else (0.2 if i == 1 else 0.1)
                layers.append(nn.Dropout(drop))
            layers.append(nn.Linear(units[-1], num_classes))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

# ========== Ensemble architectures ==========
# 5 diverse architectures (you can tweak sizes)
ARCHS = [
    (256, 128, 64),     # baseline
    (512, 256, 128),    # wide
    (128, 64),          # small
    (256, 256, 128, 64),# deep-wide
    (512, 128, 64),     # alternate shape
]

# Training config (CV inside each model)
N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Containers for ensemble
model_oof_preds = []   # list per model: oof preds shape (n_samples, n_classes) after CV
model_test_preds = []  # list per model: test probs averaged across folds (n_test, n_classes)
model_cv_scores = []   # list of mean CV (fold) macro-F1 for each model

# ========== Loop over architectures ==========
for m_idx, arch in enumerate(ARCHS, start=1):
    print("\n" + "="*40)
    print(f"Training model {m_idx} / {len(ARCHS)} with arch {arch}")
    # placeholders for this model
    oof_preds = np.zeros((len(X_fe), num_classes), dtype=np.float32)
    test_pred_accum = np.zeros((len(X_test_fe), num_classes), dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), start=1):
        print(f"\nModel {m_idx} - Fold {fold}")
        # Split
        X_tr = X_fe.iloc[tr_idx].reset_index(drop=True)
        X_val = X_fe.iloc[val_idx].reset_index(drop=True)
        X_test_now = X_test_fe.copy()

        # Scale fold-wise
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test_s = pd.DataFrame(scaler.transform(X_test_now), columns=X_test_now.columns)

        y_tr = y[tr_idx]
        y_val = y[val_idx]

        # PyTorch training
        if use_torch:
            import torch
            # build tensors/dataloaders
            Xtr_t = torch.tensor(X_tr_s.values, dtype=torch.float32)
            ytr_t = torch.tensor(y_tr, dtype=torch.long)
            Xval_t = torch.tensor(X_val_s.values, dtype=torch.float32)
            yval_t = torch.tensor(y_val, dtype=torch.long)
            Xtest_t = torch.tensor(X_test_s.values, dtype=torch.float32)

            train_ds = TensorDataset(Xtr_t, ytr_t)
            val_ds = TensorDataset(Xval_t, yval_t)
            test_ds = TensorDataset(Xtest_t)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            model = MLP(in_dim=X_tr_s.shape[1], units=list(arch)).to(device).float()
            cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=cw)
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            best_f1 = -np.inf
            no_imp = 0
            for epoch in range(1, EPOCHS+1):
                model.train()
                total_loss = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device).long()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)

                # Validation
                model.eval()
                preds_val = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device).float()
                        probs = torch.softmax(model(xb), dim=1).cpu().numpy()
                        preds_val.append(probs)
                preds_val = np.vstack(preds_val)
                val_f1 = f1_score(y_val, preds_val.argmax(axis=1), average="macro")

                if val_f1 > best_f1 + 1e-8:
                    best_f1 = val_f1
                    no_imp = 0
                    # save fold-model
                    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_m{m_idx}_fold{fold}.pt"))
                else:
                    no_imp += 1

                if no_imp >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}. Best val Macro-F1: {best_f1:.4f}")
                    break

                if epoch == 1 or epoch % 10 == 0:
                    avg_loss = total_loss / len(train_loader.dataset)
                    print(f"Epoch {epoch:03d} | Train loss {avg_loss:.4f} | Val Macro-F1 {val_f1:.4f} | Best {best_f1:.4f}")

            print(f"Model {m_idx} Fold {fold} best Macro-F1: {best_f1:.4f}")
            fold_scores.append(best_f1)

            # load best model and predict val/test
            model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"model_m{m_idx}_fold{fold}.pt"), map_location=device))
            model.to(device).float().eval()

            # OOF preds for this fold
            val_preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device).float()
                    val_preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
            val_preds = np.vstack(val_preds)
            oof_preds[val_idx] = val_preds

            # Test preds (fold-level)
            test_fold_preds = []
            with torch.no_grad():
                for xb in test_loader:
                    xb = xb[0].to(device).float()
                    test_fold_preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
            test_fold_preds = np.vstack(test_fold_preds)
            test_pred_accum += test_fold_preds

        else:
            # sklearn fallback
            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier(hidden_layer_sizes=arch, activation='relu',
                                alpha=WEIGHT_DECAY, batch_size=BATCH_SIZE, max_iter=300,
                                early_stopping=True, random_state=42)
            mlp.fit(X_tr_s.values, y_tr)
            val_probs = mlp.predict_proba(X_val_s.values)
            oof_preds[val_idx] = val_probs
            val_f1 = f1_score(y_val, val_probs.argmax(axis=1), average="macro")
            fold_scores.append(val_f1)
            print(f"Model {m_idx} Fold {fold} Macro-F1: {val_f1:.4f}")
            test_pred_accum += mlp.predict_proba(X_test_s.values)

    # after folds
    mean_cv = float(np.mean(fold_scores))
    std_cv = float(np.std(fold_scores))
    print(f"\nModel {m_idx} finished. Mean CV Macro-F1: {mean_cv:.4f}  Std: {std_cv:.4f}")

    # average test predictions across folds
    test_pred_avg = test_pred_accum / N_FOLDS

    # store per-model results
    model_oof_preds.append(oof_preds)
    model_test_preds.append(test_pred_avg)
    model_cv_scores.append(mean_cv)

# ========== Combine models into ensemble ==========
print("\n=== Combining models into final ensemble ===")
# Stack oof preds across models (num_models, n_samples, n_classes)
stack_oof = np.stack(model_oof_preds, axis=0)
# Average across models
ensemble_oof = np.mean(stack_oof, axis=0)
oof_pred_labels = ensemble_oof.argmax(axis=1)
ensemble_oof_f1 = f1_score(y, oof_pred_labels, average="macro")
print(f"Ensemble OOF Macro-F1: {ensemble_oof_f1:.4f}")

# Combine test preds
stack_test = np.stack(model_test_preds, axis=0)
ensemble_test = np.mean(stack_test, axis=0)
test_labels = ensemble_test.argmax(axis=1)
test_labels_orig = le.inverse_transform(test_labels)

# Save submission
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: test_labels_orig})
submission.to_csv(OUTPUT_SUB, index=False)
print(f"Saved ensemble submission to: {OUTPUT_SUB}")

# Print per-model CV scores
print("\nPer-model mean CV Macro-F1s:")
for i, s in enumerate(model_cv_scores, start=1):
    print(f" Model {i} (arch={ARCHS[i-1]}): {s:.4f}")
print("\nDone.")
