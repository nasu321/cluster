from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

# Load models
scaler     = joblib.load(os.path.join(MODELS, "scaler.pkl"))
kmeans     = joblib.load(os.path.join(MODELS, "kmeans.pkl"))
gmm        = joblib.load(os.path.join(MODELS, "gmm.pkl"))
agg        = joblib.load(os.path.join(MODELS, "agg.pkl"))
dbscan     = joblib.load(os.path.join(MODELS, "dbscan.pkl"))

FEATURE_ORDER = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = np.array([[float(data[f]) for f in FEATURE_ORDER]])
        X = np.log1p(X)
        X = scaler.transform(X)

        k = int(kmeans.predict(X)[0])     # ✅ UI result
        g = int(gmm.predict(X)[0])        # ✅ Backend only

        try:
            a = int(agg.fit_predict(X)[0])
        except:
            a = "N/A"

        try:
            d = int(dbscan.fit_predict(X)[0])
        except:
            d = "N/A"

        print("Debug:", {"kmeans": k, "gmm": g, "agg": a, "dbscan": d})
        return jsonify({"kmeans": k})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ Server running at http://127.0.0.1:5000")
    app.run(debug=True)
