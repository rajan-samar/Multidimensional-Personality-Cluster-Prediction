#!/usr/bin/env python3
"""
FT-Transformer + 3 MLPs ensemble (5-fold CV).
Reads:  /mnt/data/train.csv, /mnt/data/test.csv
Writes: /mnt/data/submission_ft_ensemble.csv
Saves:  /mnt/data/models_ft_ensemble/
"""
import os, warnings, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

# ---------------- Paths ----------------
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUTPUT_SUB = "submission_nn_ft.csv"
MODELS_DIR = "models_nn"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- Feature engineering (same conservative FE) ----------------
FE_INTERACTIONS = [
    ("focus_intensity", "consistency_score"),
    ("physical_activity_index", "hobby_engagement_level"),
    ("creative_expression_index", "altruism_score"),
    ("support_environment_score", "external_guidance_usage"),
]

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

# ---------------- Load ----------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

TARGET_COL = "personality_cluster"
ID_COL = "participant_id" if "participant_id" in test.columns else test.columns[0]

if TARGET_COL not in train.columns:
    raise SystemExit(f"Missing target '{TARGET_COL}' in {TRAIN_PATH}")

# base numeric features (drop id/record_code)
drop_cols = []
for c in ["participant_id","record_code"]:
    if c in train.columns:
        drop_cols.append(c)
base_features = [c for c in train.columns if c not in drop_cols + [TARGET_COL]]

# impute median globally
imputer = SimpleImputer(strategy="median")
X_base = pd.DataFrame(imputer.fit_transform(train[base_features]), columns=base_features)
X_test_base = pd.DataFrame(imputer.transform(test[base_features]), columns=base_features)

# FE
X_fe = add_simple_features(X_base, base_features)
X_test_fe = add_simple_features(X_test_base, base_features)

# labels
le = LabelEncoder()
y = le.fit_transform(train[TARGET_COL].values)
num_classes = len(le.classes_)
print("Num features after FE:", X_fe.shape[1])
print("Num classes:", num_classes, "labels:", list(le.classes_))
counts = Counter(y); n = len(y)
class_weights = np.array([n/(num_classes * counts[i]) for i in range(num_classes)], dtype=np.float32)

# ---------------- Device & seeds ----------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- Simple MLP (PyTorch) ----------------
class MLP(nn.Module):
    def __init__(self, in_dim, units):
        super().__init__()
        layers = []
        for i,u in enumerate(units):
            inp = in_dim if i==0 else units[i-1]
            layers.append(nn.Linear(inp, u))
            layers.append(nn.BatchNorm1d(u))
            layers.append(nn.ReLU())
            drop = 0.35 if i==0 else (0.2 if i==1 else 0.1)
            layers.append(nn.Dropout(drop))
        layers.append(nn.Linear(units[-1], num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------- FT-Transformer (light implementation) ----------------
class FTLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        att_out, _ = self.att(x, x, x)
        x = self.norm1(x + att_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class FTTransformer(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        # For numeric features we project each feature scalar to d_model dims (per feature token)
        self.num_features = num_features
        self.embedding = nn.Linear(1, d_model)        # shared linear projection for each feature scalar
        self.feature_norm = nn.LayerNorm(d_model)
        self.transformer_layers = nn.ModuleList([FTLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        # pooling: mean over tokens
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
    def forward(self, x):
        # x: (batch, num_features)
        B, F = x.shape
        x = x.view(B, F, 1)                 # (B, F, 1)
        x = self.embedding(x)               # (B, F, d_model)
        x = self.feature_norm(x)
        for layer in self.transformer_layers:
            x = layer(x)
        # pool across features (tokens)
        x_pooled = x.mean(dim=1)            # (B, d_model)
        out = self.head(x_pooled)
        return out

# ---------------- Ensemble setup ----------------
ARCHS = [
    (256, 128, 64),   # MLP 1
    (128, 64),        # MLP 2 (lighter)
    (512, 256, 128)   # MLP 3
]
FT_CONFIG = dict(d_model=128, n_heads=4, n_layers=2, dropout=0.15)  # conservative FT
MODELS = ["mlp1","mlp2","mlp3","ft"]

N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
GAUSS_NOISE = 0.01   # small train-time noise

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# containers
model_oof_preds = {m: np.zeros((len(X_fe), num_classes), dtype=np.float32) for m in MODELS}
model_test_preds = {m: np.zeros((len(X_test_fe), num_classes), dtype=np.float32) for m in MODELS}
model_cv_scores = {}

# ---------------- Train loop helpers ----------------
def train_model_fold(model, train_loader, val_loader, test_loader, model_name, fold):
    model = model.to(device).float()
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_f1 = -np.inf; no_imp = 0
    best_state = None
    for epoch in range(1, EPOCHS+1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            if GAUSS_NOISE > 0:
                xb = xb + torch.randn_like(xb) * GAUSS_NOISE
            yb = yb.to(device).long()
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # validate
        model.eval()
        preds_val = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                preds_val.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        preds_val = np.vstack(preds_val)
        val_f1 = f1_score(y_val_global, preds_val.argmax(axis=1), average="macro")
        if val_f1 > best_f1 + 1e-8:
            best_f1 = val_f1; no_imp = 0
            best_state = model.state_dict()
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    # save checkpoint
    ckpt = os.path.join(MODELS_DIR, f"{model_name}_fold{fold}.pt")
    torch.save(model.state_dict(), ckpt)
    # produce preds
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device).float()
            val_preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    val_preds = np.vstack(val_preds)
    test_preds_local = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb[0].to(device).float()
            test_preds_local.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    test_preds_local = np.vstack(test_preds_local)
    return best_f1, val_preds, test_preds_local

# ---------------- Main training: one model at a time ----------------
for model_key in MODELS:
    print("\n" + "="*50)
    print("Training model:", model_key)
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), start=1):
        print(f"Fold {fold} / {N_FOLDS} for model {model_key}")
        X_tr = X_fe.iloc[tr_idx].reset_index(drop=True)
        X_val = X_fe.iloc[val_idx].reset_index(drop=True)
        X_test_now = X_test_fe.copy()

        # scale fold-wise
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test_now)

        # create loaders
        Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        ytr_t = torch.tensor(y[tr_idx], dtype=torch.long)
        Xval_t = torch.tensor(X_val_s, dtype=torch.float32)
        yval_t = torch.tensor(y[val_idx], dtype=torch.long)
        Xtest_t = torch.tensor(X_test_s, dtype=torch.float32)

        train_ds = TensorDataset(Xtr_t, ytr_t)
        val_ds = TensorDataset(Xval_t, yval_t)
        test_ds = TensorDataset(Xtest_t)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # set global validation labels for helper
        global y_val_global
        y_val_global = y[val_idx]

        # instantiate model
        if model_key.startswith("mlp"):
            idx = int(model_key[-1]) - 1
            units = ARCHS[idx]
            model = MLP(in_dim=X_tr_s.shape[1], units=list(units))
        else:
            model = FTTransformer(num_features=X_tr_s.shape[1], **FT_CONFIG)

        best_f1, val_preds, test_preds_local = train_model_fold(model, train_loader, val_loader, test_loader, model_key, fold)
        print(f" Fold {fold} best val F1: {best_f1:.4f}")
        fold_scores.append(best_f1)

        # store OOF and accumulate test preds
        model_oof_preds[model_key][val_idx] = val_preds
        model_test_preds[model_key] += test_preds_local

    # average fold test preds
    model_test_preds[model_key] /= N_FOLDS
    model_cv_scores[model_key] = float(np.mean(fold_scores))
    print(f"Model {model_key} mean CV F1: {model_cv_scores[model_key]:.4f} (std {np.std(fold_scores):.4f})")

# ---------------- Combine ensemble ----------------
print("\nCombining models into final ensemble...")
# average OOF across models (only for mlp and ft keys present)
stack_oof = np.stack([model_oof_preds[k] for k in MODELS], axis=0)
ensemble_oof = stack_oof.mean(axis=0)
ensemble_oof_labels = ensemble_oof.argmax(axis=1)
ensemble_oof_f1 = f1_score(y, ensemble_oof_labels, average="macro")
print("Ensemble OOF Macro-F1:", ensemble_oof_f1)

# average test preds across models
stack_test = np.stack([model_test_preds[k] for k in MODELS], axis=0)
ensemble_test = stack_test.mean(axis=0)
test_labels = ensemble_test.argmax(axis=1)
test_labels_orig = le.inverse_transform(test_labels)

# save submission
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: test_labels_orig})
submission.to_csv(OUTPUT_SUB, index=False)
print("Saved submission to:", OUTPUT_SUB)

# print per-model CV
print("\nPer-model CV mean F1s:")
for k in MODELS:
    print(f" {k}: {model_cv_scores[k]:.4f}")
print("Done.")
