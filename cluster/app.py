from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

# --- Load all models ---
scaler = joblib.load(os.path.join(MODELS, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODELS, "kmeans.pkl"))
gmm = joblib.load(os.path.join(MODELS, "gmm.pkl"))
agg = joblib.load(os.path.join(MODELS, "agg.pkl"))
dbscan = joblib.load(os.path.join(MODELS, "dbscan.pkl"))

# --- Feature order ---
FEATURE_ORDER = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Convert input into numpy array
        X = np.array([[float(data[f]) for f in FEATURE_ORDER]])
        X = np.log1p(X)  # log-transform to normalize
        X = scaler.transform(X)  # scale the data

        # --- Perform predictions using all models ---
        try:
            k = int(kmeans.predict(X)[0])
        except:
            k = "N/A"

        try:
            g = int(gmm.predict(X)[0])
        except:
            g = "N/A"

        try:
            a = int(agg.fit_predict(X)[0])
        except:
            a = "N/A"

        try:
            d = int(dbscan.fit_predict(X)[0])
        except:
            d = "N/A"

        # Debug print
        print("Debug:", {"kmeans": k, "gmm": g, "agg": a, "dbscan": d})

        # ✅ Return DBSCAN result as the main prediction
        return jsonify({
            "dbscan_cluster": d,
            "kmeans_cluster": k,
            "gmm_cluster": g,
            "agg_cluster": a
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("✅ Server running at http://127.0.0.1:5000")
    app.run(debug=True)
