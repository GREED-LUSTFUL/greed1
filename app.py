import os, joblib, pandas as pd, requests
from flask import Flask, request, jsonify

MODEL_URL = "https://huggingface.co/greed36/intelliflight-delay-model/resolve/main/flight_delay_model.pkl"
MODEL_PATH = "flight_delay_model.pkl"

# Download at startup
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Hugging Face…")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("✅ Model downloaded")

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = joblib.load(f)

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
