import os, joblib, pandas as pd, requests
from flask import Flask, request, jsonify

# Download at startup
if not os.path.exists('flight_delay_model.pkl'):
    # shell out to your script
    os.system('./download_model.sh')

# Load model
with open('flight_delay_model.pkl', 'rb') as f:
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
