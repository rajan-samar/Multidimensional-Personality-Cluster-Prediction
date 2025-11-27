# rf_improved_fe_windows_fixed.py
"""
Fixed version: label-encoding to avoid "Mix of label input types" error.
Same behavior as rf_improved_fe_windows.py but encodes labels before training,
and decodes integer predictions back to original cluster strings for submission.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Config
# --------------------------
DATA_DIR = Path(".")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

ID_COL = "participant_id"
TARGET_COL = "personality_cluster"

RANDOM_STATE = 42
N_SPLITS = 5
BAG_MODELS = 5
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": False,
    "class_weight": "balanced",
    "n_jobs": -1
}
COARSE_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

# --------------------------
# Feature Engineering
# --------------------------
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

# --------------------------
# Load data
# --------------------------
print("Reading CSV files from:", Path.cwd())
if not (TRAIN_CSV.exists() and TEST_CSV.exists() and SAMPLE_SUB_CSV.exists()):
    raise SystemExit("train.csv, test.csv and sample_submission.csv must be in the current folder")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

if TARGET_COL not in train.columns:
    raise SystemExit(f"Target column '{TARGET_COL}' not found in train.csv")
if ID_COL not in train.columns or ID_COL not in test.columns:
    raise SystemExit(f"participant_id column missing in train/test")

test_ids = test[ID_COL].values
print("Train rows:", len(train), "Test rows:", len(test))

# --------------------------
# Prepare features + FE
# --------------------------
train_feats = train.drop(columns=[ID_COL]).copy()
y = train_feats.pop(TARGET_COL).values  # string labels
X = train_feats.copy()
X_test = test.drop(columns=[ID_COL]).copy()

X = feature_engineering(X)
X_test = feature_engineering(X_test)

# numeric-only preprocessing
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print("Using numeric columns:", num_cols)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preproc = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")

X_proc = preproc.fit_transform(X)
X_test_proc = preproc.transform(X_test)
if hasattr(X_proc, "toarray"):
    X_proc = X_proc.toarray()
if hasattr(X_test_proc, "toarray"):
    X_test_proc = X_test_proc.toarray()

# Save arrays (optional; useful for re-use)
np.save("X_proc.npy", X_proc)
np.save("X_test_proc.npy", X_test_proc)
np.save("test_ids.npy", test_ids)

# --------------------------
# LABEL ENCODING (FIX)
# --------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)          # integer labels 0..C-1
classes = le.classes_
print("Encoded classes (int -> label):")
for i, c in enumerate(classes):
    print(i, c)

# Save encoded y if needed
np.save("y_enc.npy", y_enc)

# --------------------------
# Bagged RF OOF generation (uses y_enc)
# --------------------------
n_classes = len(classes)
n_train = X_proc.shape[0]
n_test = X_test_proc.shape[0]

oof_probas_sum = np.zeros((n_train, n_classes), dtype=float)
test_probas_sum = np.zeros((n_test, n_classes), dtype=float)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print(f"Starting bagging of {BAG_MODELS} RFs with OOF generation ...")
for bag in range(BAG_MODELS):
    seed = RANDOM_STATE + bag * 13
    print(f" Bag {bag+1}/{BAG_MODELS}  (seed={seed})")
    oof_probas = np.zeros((n_train, n_classes), dtype=float)
    test_probas = np.zeros((n_test, n_classes), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_proc, y_enc), 1):
        X_tr, X_va = X_proc[tr_idx], X_proc[va_idx]
        y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

        params = dict(RF_PARAMS)
        params["random_state"] = seed + fold
        clf = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr)

        oof_probas[va_idx] = clf.predict_proba(X_va)
        test_probas += clf.predict_proba(X_test_proc) / N_SPLITS

    oof_probas_sum += oof_probas
    test_probas_sum += test_probas

oof_probas_avg = oof_probas_sum / BAG_MODELS
test_probas_avg = test_probas_sum / BAG_MODELS

# Save OOF probas
np.save("oof_probas_rf.npy", oof_probas_avg)
np.save("test_probas_rf.npy", test_probas_avg)
print("Saved oof_probas_rf.npy and test_probas_rf.npy")

# OOF baseline performance (using encoded labels)
oof_preds = oof_probas_avg.argmax(axis=1)
oof_macro = f1_score(y_enc, oof_preds, average="macro")
print("Bagged RF OOF Macro-F1 (before scaling):", round(oof_macro, 4))

# --------------------------
# Per-class multiplier search on OOF probabilities (greedy) -- works with y_enc
# --------------------------
print("Searching per-class multipliers to maximize OOF macro-F1...")

classes_int = np.arange(n_classes)
multipliers = np.ones(n_classes, dtype=float)

def apply_multipliers(probas, multipliers):
    scaled = probas * multipliers.reshape(1, -1)
    scaled = scaled / (scaled.sum(axis=1, keepdims=True) + 1e-12)
    return scaled

best_mult = multipliers.copy()
best_score = f1_score(y_enc, apply_multipliers(oof_probas_avg, best_mult).argmax(axis=1), average="macro")

for i in classes_int:
    best_local = (1.0, best_score)
    for m in COARSE_GRID:
        trial_mult = best_mult.copy()
        trial_mult[i] = m
        preds = apply_multipliers(oof_probas_avg, trial_mult).argmax(axis=1)
        sc = f1_score(y_enc, preds, average="macro")
        if sc > best_local[1]:
            best_local = (m, sc)
    best_mult[i] = best_local[0]
    best_score = best_local[1]
    print(f" class {le.inverse_transform([i])[0]} => best multiplier (coarse) = {best_local[0]}, score = {best_local[1]:.4f}")

# refine
for i in classes_int:
    center = best_mult[i]
    grid = np.linspace(max(0.5, center*0.7), center*1.4, 11)
    best_local = (center, best_score)
    for m in grid:
        trial_mult = best_mult.copy()
        trial_mult[i] = m
        preds = apply_multipliers(oof_probas_avg, trial_mult).argmax(axis=1)
        sc = f1_score(y_enc, preds, average="macro")
        if sc > best_local[1]:
            best_local = (m, sc)
    best_mult[i] = best_local[0]
    best_score = best_local[1]
    print(f" class {le.inverse_transform([i])[0]} => refined multiplier = {best_local[0]:.4f}, score = {best_local[1]:.4f}")

print("Final multipliers (label:value):", dict(zip(le.inverse_transform(classes_int), best_mult.tolist())))
print("OOF Macro-F1 after multipliers:", round(best_score, 5))
np.save("rf_oof_multipliers.npy", best_mult)

# --------------------------
# Apply multipliers to test probas, decode back to original labels and save submission
# --------------------------
scaled_test_probas = apply_multipliers(test_probas_avg, best_mult)
test_preds_int = scaled_test_probas.argmax(axis=1)
test_preds_labels = le.inverse_transform(test_preds_int)   # decode ints -> original strings

submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds_labels})
out_path = DATA_DIR / "submission_rf_improved.csv"
submission.to_csv(out_path, index=False)
print("Saved improved submission:", out_path)

# OOF classification report after scaling (decoded)
oof_preds_scaled_int = apply_multipliers(oof_probas_avg, best_mult).argmax(axis=1)
oof_preds_scaled_labels = le.inverse_transform(oof_preds_scaled_int)
print("\nOOF classification report after multiplier scaling:")
print(classification_report(y, oof_preds_scaled_labels, digits=4))
