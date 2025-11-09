import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import joblib, os

# Load dataset
df = pd.read_csv("dataset/Wholesale customers data_clustering.csv")

X = df[["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]]

# Log + scale
X_log = np.log1p(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Train models
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X_scaled)

dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan.fit(X_scaled)

# Save to /models folder
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler((1).pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(gmm, "models/gmm.pkl")
joblib.dump(agg, "models/agg.pkl")
joblib.dump(dbscan, "models/dbscan.pkl")

print("✅ All models trained & saved successfully with sklearn version 1.5.2")
