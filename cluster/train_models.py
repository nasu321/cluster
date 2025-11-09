import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import os,joblib

# ✅ Load dataset correctly from /dataset folder
DATA_PATH = os.path.join("dataset", "Wholesale customers data_clustering.csv")
df = pd.read_csv(DATA_PATH)

# ✅ Select features
X = df[["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]]

# ✅ Log + scale (same as app.py)
X_log = np.log1p(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# ✅ Train clustering models
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
gmm = GaussianMixture(n_components=3, random_state=42).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3).fit(X_scaled)
dbscan = DBSCAN(eps=1.2, min_samples=5).fit(X_scaled)

# ✅ Save them into /models folder
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(gmm, "models/gmm.pkl")
joblib.dump(agg, "models/agg.pkl")
joblib.dump(dbscan, "models/dbscan.pkl")
joblib.dump(scaler, 'models/scaler.pkl')


print("✅ Models trained & saved successfully in /models folder!")
