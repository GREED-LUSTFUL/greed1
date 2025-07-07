import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "flight_delay_model.pkl"
MODEL_URL = "https://huggingface.co/greed36/intelliflight-delay-model/resolve/main/flight_delay_model.pkl"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("⬇️ Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded.")

# Load model
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[0][1])
    return jsonify({'delayed': bool(pred), 'confidence': prob})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
