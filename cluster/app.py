from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Wholesale customers data_clustering.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    algorithm = request.form.get('algorithm')
    n_clusters = int(request.form.get('n_clusters', 3))
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = model.fit_predict(X_scaled)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=2, min_samples=5)
        df['Cluster'] = model.fit_predict(X_scaled)
    elif algorithm == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        df['Cluster'] = model.fit_predict(X_scaled)
    elif algorithm == 'agg':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        df['Cluster'] = model.fit_predict(X_scaled)
    else:
        return render_template('index.html', prediction="❌ Invalid algorithm selected")

    cluster_summary = df['Cluster'].value_counts().to_dict()
    return render_template('index.html', prediction=f"✅ {algorithm.upper()} clustering complete!", summary=cluster_summary)

if __name__ == '__main__':
    app.run(debug=True)
