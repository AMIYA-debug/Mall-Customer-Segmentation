from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open("kmeans.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


cluster_mapping = {
    0: "👉 give discounts",
    1: "👉 give loyalty points",
    2: "👉 offer premium membership",
    3: "👉 offer flash sales",
    4: "👉 offer personalized services"
}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            income = float(request.form["income"])
            spending = float(request.form["spending"])

            # Scale input
            features = np.array([[income, spending]])
            scaled_features = scaler.transform(features)

            # Predict cluster
            cluster = model.predict(scaled_features)[0]
            meaning = cluster_mapping[cluster]

            return render_template("index.html",
                                   income=income,
                                   spending=spending,
                                   cluster=cluster,
                                   meaning=meaning)
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
