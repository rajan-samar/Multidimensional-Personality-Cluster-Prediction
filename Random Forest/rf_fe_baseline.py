# rf_fe_baseline.py
"""
Final FIXED version (no OneHotEncoder at all — fully numeric, Windows-safe).
Reads: train.csv, test.csv, sample_submission.csv
Writes: submission_rf_fe.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# --------------------
# Config
# --------------------
DATA_DIR = Path(".")  # current folder
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

TARGET_COL = "personality_cluster"
ID_COL = "participant_id"

RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS = 300

# --------------------
# Feature Engineering
# --------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"focus_intensity", "consistency_score"}.issubset(df.columns):
        df["focus_consistency_mul"] = df["focus_intensity"] * df["consistency_score"]
        df["focus_consistency_div"] = df["focus_intensity"] / (df["consistency_score"] + 1e-6)

    if {"creative_expression_index", "physical_activity_index"}.issubset(df.columns):
        df["creative_physical_mul"] = df["creative_expression_index"] * df["physical_activity_index"]

    if {"support_environment_score", "external_guidance_usage"}.issubset(df.columns):
        df["support_guidance_ratio"] = df["support_environment_score"] / (df["external_guidance_usage"] + 1e-6)

    if "age_group" in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df["age_group"]):
                df["age_group_bin"] = pd.cut(df["age_group"], bins=[-1,1,2,3,4,10], labels=False)
        except:
            pass

    return df

# --------------------
# Load data
# --------------------
print(f"Reading CSVs from: {Path.cwd()}")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if TARGET_COL not in train.columns:
    raise ValueError(f"{TARGET_COL} not in train.csv")

test_ids = test[ID_COL].values

train_feats = train.drop(columns=[ID_COL])
y = train_feats.pop(TARGET_COL).values
X = train_feats.copy()
X_test = test.drop(columns=[ID_COL]).copy()

# Apply FE
X_fe = feature_engineering(X)
X_test_fe = feature_engineering(X_test)

# --------------------
# Only numeric preprocessing
# --------------------
num_cols = X_fe.select_dtypes(include=[float, int]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns: NONE — OneHotEncoder removed completely")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
], remainder="drop")

print("Fitting preprocessor...")
X_proc = preprocessor.fit_transform(X_fe)
X_test_proc = preprocessor.transform(X_test_fe)

# --------------------
# RandomForest + CV
# --------------------
print("Running Stratified K-Fold CV with RandomForest...")

rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_scores = []
all_true = []
all_pred = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_proc, y), 1):
    X_tr, X_val = X_proc[tr_idx], X_proc[va_idx]
    y_tr, y_val = y[tr_idx], y[va_idx]

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)

    f1 = f1_score(y_val, preds, average="macro")
    fold_scores.append(f1)

    all_true.extend(y_val.tolist())
    all_pred.extend(preds.tolist())

    print(f"Fold {fold} Macro-F1: {f1:.4f}")

print("\n=== CV RESULTS ===")
print("Mean Macro-F1:", np.mean(fold_scores))
print("Std  Macro-F1:", np.std(fold_scores))
print("\nClassification report:")
print(classification_report(all_true, all_pred, digits=4))

# --------------------
# Final model
# --------------------
print("Training final model...")
final_rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)
final_rf.fit(X_proc, y)
test_preds = final_rf.predict(X_test_proc)

# --------------------
# Submission
# --------------------
sub = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: test_preds
})

out_path = Path.cwd() / "submission_rf_fe.csv"
sub.to_csv(out_path, index=False)

print(f"\nSaved submission file: {out_path}")
print("Upload to Kaggle. DONE.")
