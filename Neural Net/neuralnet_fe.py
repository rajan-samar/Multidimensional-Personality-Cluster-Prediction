#!/usr/bin/env python3
"""
train_nn_fe.py

Neural-net-only pipeline with feature engineering (no plots).
- Place train.csv and test.csv in the same folder and run:
    python train_nn_fe.py
Outputs:
- submission_nn.csv
- models_nn/model_fold{fold}.pt
Prints Macro-F1 per fold to terminal.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

# ---------------------
# Paths (local)
# ---------------------
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUTPUT_SUB = "submission_nn.csv"
MODELS_DIR = "models_nn"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------
# Load data
# ---------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

id_col = "participant_id" if "participant_id" in test.columns else test.columns[0]
test_ids = test[id_col].copy()

target_col = "personality_cluster"
if target_col not in train.columns:
    raise ValueError(f"Target '{target_col}' not found in train.csv")

# Drop identifier-like columns from features (but keep id for submission)
drop_cols = []
for c in ["participant_id", "record_code"]:
    if c in train.columns:
        drop_cols.append(c)

base_feature_cols = [c for c in train.columns if c not in drop_cols + [target_col]]
print("Base feature cols:", base_feature_cols)

# ---------------------
# Feature engineering function
# ---------------------
def add_features(df):
    """
    Input: DataFrame with base_feature_cols
    Returns: DataFrame extended with engineered features
    """
    df_fe = df.copy()

    # Interaction features (products)
    interactions = [
        ("focus_intensity", "consistency_score"),
        ("physical_activity_index", "hobby_engagement_level"),
        ("creative_expression_index", "altruism_score"),
        ("support_environment_score", "external_guidance_usage"),
    ]
    for a, b in interactions:
        if a in df_fe.columns and b in df_fe.columns:
            df_fe[f"{a}_x_{b}"] = df_fe[a] * df_fe[b]

    # Squared features (second-degree polynomial)
    for c in base_feature_cols:
        if c in df_fe.columns:
            df_fe[f"{c}_sq"] = df_fe[c] * df_fe[c]

    # sqrt-abs transforms to reduce skew (safe transform)
    for c in base_feature_cols:
        if c in df_fe.columns:
            df_fe[f"sqrt_abs_{c}"] = np.sqrt(np.abs(df_fe[c].fillna(0)) + 1.0)

    # log1p on positive-ish features (guarded)
    for c in base_feature_cols:
        if c in df_fe.columns:
            # only apply log1p if minimal value > -0.9 to avoid invalid log
            minv = df_fe[c].min()
            if minv > -0.9:
                df_fe[f"log1p_{c}"] = np.log1p(df_fe[c].fillna(0))
    return df_fe

# ---------------------
# Pre-impute -> FE -> scale pipeline helpers
# ---------------------
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
le = LabelEncoder()

# We'll build train/test features using FE after imputation
# Note: we'll perform fold-wise scaling inside CV to avoid leakage.

# 1) Impute base features (train & test) with median computed from full train (safe before FE)
X_base = pd.DataFrame(imputer.fit_transform(train[base_feature_cols]), columns=base_feature_cols)
X_test_base = pd.DataFrame(imputer.transform(test[base_feature_cols]), columns=base_feature_cols)

# 2) Add FE (creates new columns)
X_fe = add_features(X_base)
X_test_fe = add_features(X_test_base)

print("FE features count:", X_fe.shape[1], "columns example:", X_fe.columns.tolist()[:12])

# 3) Prepare labels
y = le.fit_transform(train[target_col].values)
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))
print("Num features after FE:", X_fe.shape[1])

# Class weights (float32)
counts = Counter(y)
n = len(y)
class_weights = np.array([n / (num_classes * counts[i]) for i in range(num_classes)], dtype=np.float32)
print("Class distribution:", counts)
print("Class weights:", class_weights)

# ---------------------
# Try to import torch (prefer), else sklearn
# ---------------------
use_torch = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    # seeds
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    use_torch = False
    device = "cpu"

print("Use PyTorch:", use_torch, "Device:", device)

# ---------------------
# Define torch Dataset & model if torch available
# ---------------------
if use_torch:
    class TabDataset(Dataset):
        def __init__(self, X, y=None):
            # Ensure float32 and avoid NaNs
            self.X = np.asarray(X, dtype=np.float32)
            self.y = None if y is None else np.asarray(y, dtype=np.int64)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            if self.y is None:
                return self.X[idx]
            return self.X[idx], self.y[idx]

    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64,out_dim)
            )
        def forward(self, x):
            return self.net(x)

else:
    from sklearn.neural_network import MLPClassifier

# ---------------------
# Training hyperparams
# ---------------------
N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42

# ---------------------
# Cross-validation with fold-wise scaling (no leakage)
# ---------------------
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros((len(X_fe), num_classes), dtype=np.float32)
test_preds = np.zeros((len(X_test_fe), num_classes), dtype=np.float32)
fold_scores = []

feature_names = X_fe.columns.tolist()

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), start=1):
    print(f"\n--- Fold {fold} ---")
    X_tr = X_fe.iloc[tr_idx].reset_index(drop=True)
    X_val = X_fe.iloc[val_idx].reset_index(drop=True)
    X_test_now = X_test_fe.copy()

    # Fit scaler on training fold only
    scaler_fold = StandardScaler()
    X_tr_scaled = pd.DataFrame(scaler_fold.fit_transform(X_tr), columns=feature_names)
    X_val_scaled = pd.DataFrame(scaler_fold.transform(X_val), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler_fold.transform(X_test_now), columns=feature_names)

    y_tr = y[tr_idx]
    y_val = y[val_idx]

    if use_torch:
        train_ds = TabDataset(X_tr_scaled.values, y_tr)
        val_ds = TabDataset(X_val_scaled.values, y_val)
        test_ds = TabDataset(X_test_scaled.values, None)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = MLP(in_dim=X_tr_scaled.shape[1], out_dim=num_classes).to(device).float()
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_f1 = -np.inf
        no_improve = 0

        for epoch in range(1, EPOCHS+1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

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
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_fold{fold}.pt"))
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val macro-F1: {best_f1:.4f}")
                break

            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Val Macro-F1: {val_f1:.4f} | Best: {best_f1:.4f}")

        print(f"Fold {fold} best Macro-F1: {best_f1:.4f}")
        fold_scores.append(best_f1)

        # Load best model and get OOF + test preds
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"model_fold{fold}.pt"), map_location=device))
        model.to(device).float().eval()

        # OOF preds
        preds_val = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device).float()
                preds_val.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        oof_preds[val_idx] = np.vstack(preds_val)

        # Test preds (accumulate)
        preds_test = []
        with torch.no_grad():
            for xb in test_loader:
                xb = xb.to(device).float()
                preds_test.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        test_preds += np.vstack(preds_test)

    else:
        # sklearn fallback: fit on scaled fold data
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),
                            activation='relu',
                            alpha=WEIGHT_DECAY,
                            batch_size=BATCH_SIZE,
                            max_iter=300,
                            early_stopping=True,
                            random_state=SEED)
        mlp.fit(X_tr_scaled.values, y_tr)
        val_probs = mlp.predict_proba(X_val_scaled.values)
        oof_preds[val_idx] = val_probs
        val_f1 = f1_score(y_val, val_probs.argmax(axis=1), average="macro")
        fold_scores.append(val_f1)
        print(f"Fold {fold} Macro-F1: {val_f1:.4f}")
        test_preds += mlp.predict_proba(X_test_scaled.values)

# ---------------------
# Final metrics & submission
# ---------------------
print("\n=== CV RESULTS ===")
for i, sc in enumerate(fold_scores, 1):
    print(f"Fold {i} Macro-F1: {sc:.4f}")
print("Mean Macro-F1:", np.mean(fold_scores), "Std:", np.std(fold_scores))

# overall OOF
oof_labels = oof_preds.argmax(axis=1)
overall_oof = f1_score(y, oof_labels, average="macro")
print("OOF Macro-F1 (overall):", overall_oof)

# prepare submission
test_preds /= N_FOLDS
test_labels = test_preds.argmax(axis=1)
test_labels_orig = le.inverse_transform(test_labels)

submission = pd.DataFrame({id_col: test_ids, target_col: test_labels_orig})
submission.to_csv(OUTPUT_SUB, index=False)
print(f"\nSaved submission to: {OUTPUT_SUB} (rows: {len(submission)})")
