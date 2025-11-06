# 🧩 Wholesale Customers Clustering System  
### 🧑‍💻 Developed by: **Mohammad Naseem**

This project performs **Unsupervised Learning** on the **Wholesale Customers dataset** to group customers into clusters using multiple **Clustering Algorithms**.  
It is deployed as a **Flask Web Application** that allows you to choose different algorithms and view the clustering summary interactively.

---

## 🎯 Project Overview
The **Wholesale Customers Clustering System** segments customers into meaningful groups based on their purchasing behavior.  
It applies several clustering algorithms and compares how they group the same dataset differently.

---

## 🧠 Algorithms Used
| Algorithm | Description |
|------------|--------------|
| **1️⃣ K-Means** | Centroid-based clustering that minimizes intra-cluster variance. |
| **2️⃣ DBSCAN** | Density-based clustering that identifies noise and arbitrary shaped clusters. |
| **3️⃣ Gaussian Mixture Model (GMM)** | Probabilistic model assuming data is generated from a mixture of Gaussians. |
| **4️⃣ Agglomerative Clustering** | Hierarchical clustering method building nested clusters by merging similar ones. |
| **5️⃣ StandardScaler / MinMaxScaler** | Used to normalize the data before clustering. |

---

## 📁 Project Structure

wholesale_clustering/
│
├── app.py # Flask backend
├── clustering_train.py # Model training & visualization
├── dataset/
│ └── Wholesale customers data_clustering.csv
├── templates/
│ └── index.html # Web interface
├── static/
│ └── css/
│ └── style.css # Styling
├── screenshots/
│ └── clustering_output.png # Example output image
└── README.md
## 📈 Evaluation Metrics

Although clustering is unsupervised, performance can be evaluated using metrics such as:

Silhouette Score

Davies-Bouldin Index

Calinski-Harabasz Score

Cluster Distribution Visualization

## 🧠 Technologies Used
| Category                 | Technology                              |
| ------------------------ | --------------------------------------- |
| **Programming Language** | Python                                  |
| **Libraries**            | scikit-learn, pandas, numpy, flask      |
| **Algorithms**           | KMeans, DBSCAN, GMM, Agglomerative      |
| **Web Framework**        | Flask                                   |
| **Frontend**             | HTML5, CSS3                             |
| **Dataset**              | Wholesale customers data_clustering.csv |


