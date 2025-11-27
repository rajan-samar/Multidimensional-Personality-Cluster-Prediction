#!/usr/bin/env python3
"""
Neural-net only training pipeline (PyTorch preferred, sklearn fallback).
Run this in terminal:
    python train_nn.py

Expected files in same folder:
    train.csv
    test.csv

Outputs:
    submission_nn.csv
    models_nn/model_foldX.pt
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
# Local paths (same folder)
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

id_col = "participant_id"
test_ids = test[id_col]

drop_cols = ["participant_id", "record_code"]
target_col = "personality_cluster"

feature_cols = [c for c in train.columns if c not in drop_cols + [target_col]]
print("Feature columns:", feature_cols)

# ---------------------
# Preprocessing
# ---------------------
X = train[feature_cols].copy()
y_raw = train[target_col].copy()
X_test = test[feature_cols].copy()

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))

# Class weights
counts = Counter(y)
n = len(y)
class_weights = np.array([n / (num_classes * counts[i]) for i in range(num_classes)], dtype=np.float32)
print("Class weights:", class_weights)

# ---------------------
# Try PyTorch
# ---------------------
use_torch = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except:
    use_torch = False
    device = "cpu"

print("Using PyTorch:", use_torch, "| Device:", device)

# ---------------------
# Torch classes
# ---------------------
if use_torch:
    class TabDS(Dataset):
        def __init__(self, X, y=None):
            self.X = np.asarray(X, dtype=np.float32)
            self.y = None if y is None else np.asarray(y, dtype=np.int64)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            if self.y is None: return self.X[idx]
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
                nn.Linear(64, out_dim)
            )
        def forward(self, x):
            return self.net(x)

# ---------------------
# sklearn fallback
# ---------------------
if not use_torch:
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
# Cross-validation loop
# ---------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros((len(X), num_classes), dtype=np.float32)
test_preds = np.zeros((len(X_test), num_classes), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n========== Fold {fold} ==========")

    X_tr, X_val = X.values[tr_idx], X.values[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    if use_torch:
        train_ds = TabDS(X_tr, y_tr)
        val_ds = TabDS(X_val, y_val)
        test_ds = TabDS(X_test.values, None)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = MLP(len(feature_cols), num_classes).to(device).float()

        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_f1 = -1
        no_imp = 0

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
            preds = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device).float()
                    prob = torch.softmax(model(xb), dim=1).cpu().numpy()
                    preds.append(prob)
            preds = np.vstack(preds)
            f1 = f1_score(y_val, preds.argmax(axis=1), average="macro")

            if f1 > best_f1:
                best_f1 = f1
                no_imp = 0
                torch.save(model.state_dict(), f"{MODELS_DIR}/model_fold{fold}.pt")
            else:
                no_imp += 1

            if no_imp >= PATIENCE:
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: val_f1 = {f1:.4f} | best = {best_f1:.4f}")

        print(f"Fold {fold} F1 = {best_f1:.4f}")
        fold_scores.append(best_f1)

        # Load best model & predict
        model.load_state_dict(torch.load(f"{MODELS_DIR}/model_fold{fold}.pt", map_location=device))
        model.to(device).float().eval()

        # OOF
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device).float()
                preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        oof_preds[val_idx] = np.vstack(preds)

        # Test
        preds = []
        with torch.no_grad():
            for xb in test_loader:
                xb = xb.to(device).float()
                preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        test_preds += np.vstack(preds)

    else:
        # sklearn fallback
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,128,64),
            activation="relu",
            alpha=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
            max_iter=300,
            early_stopping=True,
            random_state=SEED
        )
        mlp.fit(X_tr, y_tr)
        val_prob = mlp.predict_proba(X_val)
        oof_preds[val_idx] = val_prob
        f1 = f1_score(y_val, val_prob.argmax(axis=1), average="macro")
        fold_scores.append(f1)
        print(f"Fold {fold} F1 = {f1:.4f}")

        test_preds += mlp.predict_proba(X_test.values)

# ---------------------
# Final logging
# ---------------------
print("\n===== CV SUMMARY =====")
for i, f in enumerate(fold_scores, 1):
    print(f"Fold {i}: {f:.4f}")

print("Mean F1:", np.mean(fold_scores))
print("Std F1:", np.std(fold_scores))

# Overall OOF F1
oof_labels = oof_preds.argmax(axis=1)
overall = f1_score(y, oof_labels, average="macro")
print("Overall OOF Macro-F1:", overall)

# ---------------------
# Submission
# ---------------------
test_preds /= N_FOLDS
test_labels = test_preds.argmax(axis=1)
test_labels_orig = le.inverse_transform(test_labels)

sub = pd.DataFrame({
    id_col: test_ids,
    target_col: test_labels_orig
})

sub.to_csv(OUTPUT_SUB, index=False)
print(f"\nSaved submission to: {OUTPUT_SUB}")
