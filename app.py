from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# -------------------------------------------------
# Training data exactly from the supplied notebook
# -------------------------------------------------
celsius_q = np.array(
    [-40, -10, 0, 8, 10, 15, 22, 50, 20, 38],
    dtype=float
)

fahrenheit_a = np.array(
    [-40.0, 14.0, 32.0, 46.4, 50.0, 59.0, 71.6, 122.0, 68.0, 100.4],
    dtype=float
)

# -------------------------------------------------
# Exact final neural-network model from the notebook
# Dense(4) -> Dense(4) -> Dense(1)
# -------------------------------------------------
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0, l1, l2])

model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1)
)

# Exact final training setting from the notebook
model.fit(
    celsius_q,
    fahrenheit_a,
    epochs=800,
    verbose=False
)

# -------------------------------------------------
# Home page
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------
# Prediction API
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if data is None:
            data = request.form

        celsius = float(data.get("celsius"))

        prediction = model.predict(
            np.array([[celsius]], dtype=float),
            verbose=0
        )

        fahrenheit = float(prediction[0][0])

        return jsonify({
            "celsius": round(celsius, 2),
            "fahrenheit": round(fahrenheit, 2)
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Please enter a valid Celsius temperature."
        }), 400


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "TensorFlow Dense Neural Network 4-4-1",
        "epochs": 800
    })


# -------------------------------------------------
# Run locally
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
