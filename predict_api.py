from flask import Flask, request, jsonify
import joblib
import pandas as pd
import gzip

app = Flask(__name__)

# ✅ Load the model from a compressed gzip file
with gzip.open('flight_delay_model.pkl.gz', 'rb') as f:
    model = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[0][1])
    return jsonify({'delayed': bool(pred), 'confidence': prob})

if __name__ == '__main__':
    app.run(port=5001)
