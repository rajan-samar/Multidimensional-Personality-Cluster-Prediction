"""
soup_submissions.py
Blend multiple submission CSVs into a "soup" (averaged probabilities) and export final labels.

Place this file in the folder with:
 - train.csv (used to discover class names / labels order)
 - test.csv (used to align participant_id)
 - submission CSVs (any number). They can be:
    * probability submissions with one column per class (class names should match train labels), OR
    * label-only submissions with a column named 'personality_cluster' (or TARGET_COL below).

Output: submission_soup.csv

Windows-friendly. Use PowerShell to run.

Examples:
  python soup_submissions.py
  python soup_submissions.py --files s1.csv s2.csv --weights 1 0.5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

# --------------------
ID_COL = "participant_id"
TARGET_COL = "personality_cluster"
DATA_DIR = Path(".")
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE  = DATA_DIR / "test.csv"
SAMPLE_FILE = DATA_DIR / "sample_submission.csv"

# --------------------
def find_submission_files():
    # prefer files named submission*.csv or any csv excluding train/test/sample and this script
    files = sorted([p for p in DATA_DIR.glob("*.csv")])
    candidates = []
    for p in files:
        name = p.name.lower()
        if name in {TRAIN_FILE.name.lower(), TEST_FILE.name.lower(), SAMPLE_FILE.name.lower(), Path(__file__).name.lower()}:
            continue
        # skip obvious intermediate files you don't want (optional)
        if "oof" in name or "probas" in name:
            continue
        candidates.append(p)
    # If there is sample_submission only (unlikely), return empty
    return candidates

def load_classes_from_train(train_path):
    df = pd.read_csv(train_path)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"train.csv missing expected target column '{TARGET_COL}'")
    le = LabelEncoder()
    le.fit(df[TARGET_COL].astype(str).values)
    classes = list(le.classes_)
    return le, classes

def read_submission_as_proba(path, classes, test_df, le):
    """
    Return (participant_ids, prob_matrix) where prob_matrix shape = (n_test, n_classes).
    If CSV has class columns, use them (matching class names).
    If CSV has label column, convert to one-hot using classes order.
    """
    df = pd.read_csv(path)
    # ensure participant id present, else assume same order as test
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)
    else:
        # no id column - assume same order as provided test.csv
        df.index = test_df[ID_COL].values

    # detect probability columns: intersection with classes
    colset = set(df.columns.astype(str))
    class_cols = [c for c in classes if c in colset]
    if len(class_cols) == len(classes):
        # use these columns in class order
        probs = df[class_cols].values
        # reorder columns to classes order just in case
        probs = pd.DataFrame(probs, index=df.index, columns=class_cols)[classes].values
        return df.index.to_numpy(), probs
    else:
        # fallback: if there is TARGET_COL (label), convert to one-hot
        if TARGET_COL in df.columns:
            labels = df[TARGET_COL].astype(str).values
            # map labels -> indices using label encoder (handles unknown)
            try:
                idxs = le.transform(labels)
            except Exception:
                # if labels contain unseen classes, map carefully
                idxs = []
                for s in labels:
                    if s in classes:
                        idxs.append(classes.index(s))
                    else:
                        # Unknown label -> assign uniform distribution (very unlikely)
                        idxs.append(-1)
                idxs = np.array(idxs, dtype=int)
            n = len(labels)
            probs = np.zeros((n, len(classes)), dtype=float)
            for i, ix in enumerate(idxs):
                if ix >= 0:
                    probs[i, ix] = 1.0
                else:
                    probs[i, :] = 1.0 / len(classes)
            return df.index.to_numpy(), probs
        else:
            raise ValueError(f"Submission {path} has neither class probability columns nor '{TARGET_COL}' label column.")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="List of submission CSV files to blend. If omitted, auto-discovers CSVs.")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for each file (same length as files). Default=equal weights.")
    parser.add_argument("--out", default="submission_soup.csv", help="Output filename.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # check train/test
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise SystemExit("train.csv and test.csv must exist in the working folder.")

    le, classes = load_classes_from_train(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    n_test = test_df.shape[0]

    # find files
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = find_submission_files()
    if len(files) == 0:
        raise SystemExit("No submission CSVs found. Provide --files or place submission CSVs in the folder.")

    # Load each submission as probabilities
    proba_list = []
    ids_list  = []
    good_files = []
    for f in files:
        try:
            ids, probs = read_submission_as_proba(f, classes, test_df, le)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            continue
        # align to test_df order by participant_id
        if not np.array_equal(ids, test_df[ID_COL].values):
            # reorder
            if args.verbose:
                print(f"Aligning {f.name} by {ID_COL} to test order.")
            probs_df = pd.DataFrame(probs, index=ids, columns=classes)
            probs_df = probs_df.reindex(test_df[ID_COL].values).fillna(1.0/len(classes))
            probs = probs_df.values
        proba_list.append(probs)
        ids_list.append(test_df[ID_COL].values)
        good_files.append(f)

    m = len(proba_list)
    if m == 0:
        raise SystemExit("No valid submission files to blend after scanning.")

    # weights
    if args.weights:
        if len(args.weights) != m:
            raise SystemExit("Number of weights must match number of files used.")
        weights = np.array(args.weights, dtype=float)
    else:
        weights = np.ones(m, dtype=float)

    # normalize weights optionally (not necessary)
    # weights = weights / weights.sum()

    if args.verbose:
        print("Files used for blending:")
        for f, w in zip(good_files, weights):
            print(" ", f.name, "weight", w)

    # build weighted proba
    final_proba = np.zeros((n_test, len(classes)), dtype=float)
    for w, p in zip(weights, proba_list):
        final_proba += w * p

    # normalize rows
    final_proba = final_proba / (final_proba.sum(axis=1, keepdims=True) + 1e-12)

    # final preds
    final_idx = np.argmax(final_proba, axis=1)
    final_labels = le.inverse_transform(final_idx)

    out_df = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: final_labels})
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Saved blended submission: {out_path.resolve()}")
    print("Files blended:")
    for f in good_files:
        print("  -", f.name)
    print("Weights:", weights.tolist())

if __name__ == "__main__":
    main(sys.argv[1:])
