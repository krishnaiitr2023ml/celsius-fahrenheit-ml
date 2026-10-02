from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Training data taken from the supplied notebook
celsius_q = np.array([-40, -10, 0, 8, 10, 15, 22, 50, 20, 38], dtype=float)
fahrenheit_a = np.array([-40.0, 14.0, 32.0, 46.4, 50.0, 59.0, 71.6, 122.0, 68.0, 100.4], dtype=float)

# Learn the best-fit linear model F = w*C + b from the same data.
# This keeps the deployed app lightweight and avoids loading TensorFlow on Render.
weight, bias = np.polyfit(celsius_q, fahrenheit_a, 1)

def predict_fahrenheit(celsius):
    return weight * float(celsius) + bias

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form
        celsius = float(data.get("celsius"))
        fahrenheit = predict_fahrenheit(celsius)
        return jsonify({
            "celsius": round(celsius, 4),
            "fahrenheit": round(float(fahrenheit), 2)
        })
    except (TypeError, ValueError):
        return jsonify({"error": "Please enter a valid Celsius value."}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
