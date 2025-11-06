import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
BASE_DIR = os.path.dirname(__file__)
# Try multiple locations for the dataset (project root or a dataset/ folder)
possible_paths = [
    os.path.join(BASE_DIR, "dataset", "Wholesale customers data_clustering.csv"),
    os.path.join(BASE_DIR, "Wholesale customers data_clustering.csv"),
    os.path.join(BASE_DIR, "data", "Wholesale customers data_clustering.csv"),
]
for p in possible_paths:
    if os.path.exists(p):
        DATA_PATH = p
        break
else:
    raise FileNotFoundError(f"Dataset not found. Checked: {possible_paths}")

MODELS_DIR = os.path.join(BASE_DIR, "models")

SPEND_COLS = ["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
df = pd.read_csv(DATA_PATH)

def _load_model_file(fname, required=False):
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required model file not found: {path}")
        else:
            return None
    return joblib.load(path)

# scaler is required to transform the data
scaler = _load_model_file("scaler.pkl", required=True)
# other models are optional but useful
kmeans = _load_model_file("kmeans.pkl", required=False)
gmm = _load_model_file("gmm.pkl", required=False)
agg = _load_model_file("agg.pkl", required=False)

X = df[SPEND_COLS].copy().fillna(0)
X_log = np.log1p(X)
X_scaled = scaler.transform(X_log)

# Predict using available models (skip if not present)
k_labels = kmeans.predict(X_scaled) if kmeans is not None and hasattr(kmeans, "predict") else None
g_labels = gmm.predict(X_scaled) if gmm is not None and hasattr(gmm, "predict") else None

# AgglomerativeClustering does not implement `predict`; prefer using stored labels_ if available
a_labels = None
if agg is not None:
    if hasattr(agg, "labels_") and getattr(agg, "labels_") is not None:
        a_labels = agg.labels_
    else:
        # try to compute labels from the model (fit_predict) as a last resort
        try:
            a_labels = agg.fit_predict(X_scaled)
        except Exception:
            a_labels = None

df_out = df.copy()
df_out["kmeans_cluster"] = k_labels
df_out["gmm_cluster"] = g_labels
if a_labels is not None:
    df_out["agg_cluster"] = a_labels
out_path = os.path.join(MODELS_DIR, "clustered_dataset.csv")
df_out.to_csv(out_path, index=False)
print("Saved clustered dataset to", out_path)
