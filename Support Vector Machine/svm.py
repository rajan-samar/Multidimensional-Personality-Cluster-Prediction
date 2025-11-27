# svm_baseline.py
"""
SVM baseline for the personality-cluster multiclass problem.

Place this script in the folder with:
 - train.csv
 - test.csv
 - sample_submission.csv

Outputs:
 - oof_probas_svm.npy
 - test_probas_svm.npy
 - submission_svm.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import SVC
from scipy.stats import rankdata

# -------- Config --------
DATA_DIR = Path(".")
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE  = DATA_DIR / "test.csv"
SAMPLE_FILE = DATA_DIR / "sample_submission.csv"

ID_COL = "participant_id"
TARGET_COL = "personality_cluster"

RANDOM_STATE = 42
N_FOLDS = 5

# -------- Helpers --------
def safe_read_csv(p: Path):
    if not p.exists():
        raise SystemExit(f"Missing file: {p.resolve()}")
    return pd.read_csv(p)

def numeric_columns_auto(df, excluded=set()):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in nums if c not in excluded]

# -------- Load data --------
print("Loading data...")
train = safe_read_csv(TRAIN_FILE)
test = safe_read_csv(TEST_FILE)
sample = safe_read_csv(SAMPLE_FILE)
print(f"Train rows: {train.shape[0]}, Test rows: {test.shape[0]}")

# -------- Basic feature selection & safe fills --------
excluded = {ID_COL, TARGET_COL, "record_code"}
base_num = numeric_columns_auto(train, excluded)
if len(base_num) == 0:
    base_num = [c for c in train.columns if c not in excluded]

print("Numeric features used:", base_num)

def build_features(df):
    df = df.copy()
    # Ensure base numeric present
    for c in base_num:
        if c not in df.columns:
            df[c] = 0.0

    # Some main behavioral features for interactions if present
    inter = []
    for c in [
        "focus_intensity",
        "consistency_score",
        "support_environment_score",
        "creative_expression_index",
        "physical_activity_index",
    ]:
        if c in df.columns:
            inter.append(c)

    # Limited pairwise interactions to avoid explosion
    for i in range(len(inter)):
        for j in range(i + 1, min(i + 3, len(inter))):
            c1, c2 = inter[i], inter[j]
            a = df[c1].fillna(0.0).astype(float)
            b = df[c2].fillna(0.0).astype(float)
            df[f"{c1}_mul_{c2}"] = a * b
            df[f"{c1}_div_{c2}"] = a / (b + 1e-6)

    # Simple power transforms
    for c in inter:
        arr = df[c].fillna(0.0).astype(float)
        df[f"{c}_sq"] = arr ** 2
        df[f"{c}_sqrt"] = np.sqrt(np.clip(arr, 0.0, None))

    # Rank-normalized versions of base numeric features
    for c in base_num:
        arr = df[c].fillna(df[c].median()).to_numpy()
        ranks = rankdata(arr)
        df[f"{c}_rank"] = ranks / (len(arr) + 1.0)

    df = df.fillna(0.0)
    return df

print("Building features...")
train_fe = build_features(train)
test_fe = build_features(test)

drop_cols = {ID_COL, TARGET_COL, "record_code"}
features = [c for c in train_fe.columns if c not in drop_cols]
print("Total features for SVM:", len(features))

# -------- Scale features (fit only on train) --------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_fe[features].values)
X_test = scaler.transform(test_fe[features].values)

# -------- Labels --------
le = LabelEncoder()
y = le.fit_transform(train[TARGET_COL].astype(str).values)
classes = le.classes_
n_classes = len(classes)
print("Classes:", list(classes))

n_train = X_train.shape[0]
n_test = X_test.shape[0]

oof_probas = np.zeros((n_train, n_classes), dtype=float)
test_probas = np.zeros((n_test, n_classes), dtype=float)

# -------- CV training with SVM (RBF kernel) --------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print("\nTraining SVM (RBF, probability=True) with StratifiedKFold...")
t0 = time.time()
fold_no = 0

for tr_idx, va_idx in skf.split(X_train, y):
    fold_no += 1
    print(f"\n--- Fold {fold_no}/{N_FOLDS} ---")
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # SVM classifier
    svm = SVC(
        kernel="rbf",
        C=2.0,              # you can try 1.0, 2.0, 4.0, etc later
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE
    )

    svm.fit(X_tr, y_tr)

    proba_va = svm.predict_proba(X_va)
    proba_test_fold = svm.predict_proba(X_test)

    oof_probas[va_idx] = proba_va
    test_probas += proba_test_fold / N_FOLDS

    va_preds = proba_va.argmax(axis=1)
    f1 = f1_score(y_va, va_preds, average="macro")
    print(f"Fold {fold_no} Macro-F1: {f1:.4f}")

print(f"\nTraining finished in {time.time() - t0:.1f}s")

# -------- OOF evaluation --------
oof_preds = oof_probas.argmax(axis=1)
oof_macro = f1_score(y, oof_preds, average="macro")
print("\n=== OOF RESULTS (SVM) ===")
print("Overall OOF Macro-F1:", oof_macro)
print(
    classification_report(
        le.inverse_transform(y),
        le.inverse_transform(oof_preds),
        digits=4
    )
)

# Save OOF/test prob arrays
np.save(DATA_DIR / "oof_probas_svm.npy", oof_probas)
np.save(DATA_DIR / "test_probas_svm.npy", test_probas)
print("Saved oof_probas_svm.npy and test_probas_svm.npy")

# -------- Create submission --------
test_preds = test_probas.argmax(axis=1)
test_labels = le.inverse_transform(test_preds.astype(int))
submission = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET_COL: test_labels})
out_path = DATA_DIR / "submission_svm.csv"
submission.to_csv(out_path, index=False)
print("Saved submission:", out_path)

print("\nDone. Submit submission_svm.csv to Kaggle and tell me the score + OOF F1 if you want to tune C/gamma next.")
