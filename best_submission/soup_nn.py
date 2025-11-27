import pandas as pd
import numpy as np

# strongest submissions only (4-best)
files = [
    "submission_meta_tuned.csv",
    "submission_meta_blend.csv",
    "submission_stacked.csv",
    "submission_superstack_blend.csv"
]

print("Using files:")
for f in files:
    print(" -", f)

# read all
dfs = [pd.read_csv(f) for f in files]

# assume same row order / same participant_id order
ids = dfs[0]["participant_id"].values

classes = ["Cluster_A", "Cluster_B", "Cluster_C", "Cluster_D", "Cluster_E"]
c2i = {c: i for i, c in enumerate(classes)}

# (n_samples, n_classes)
probs = np.zeros((len(ids), len(classes)))

# convert each submission's labels to one-hot → accumulate → average
for df in dfs:
    p = np.zeros_like(probs)
    lab = df["personality_cluster"].values
    idx = np.array([c2i[x] for x in lab])
    p[np.arange(len(ids)), idx] = 1
    probs += p

probs /= len(dfs)   # average soup

# final hard predictions
final_labels = [classes[i] for i in probs.argmax(axis=1)]

out = pd.DataFrame({
    "participant_id": ids,
    "personality_cluster": final_labels
})

out.to_csv("submission_final_soup.csv", index=False)
print("\nWrote submission_final_soup.csv")
