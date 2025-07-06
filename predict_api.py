from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('flight_delay_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Expect keys: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, DISTANCE, SCHEDULED_DEPARTURE
    df = pd.DataFrame([data])
    # No need to re-encode if you send already encoded features, or replicate LabelEncoder logic here.
    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[0][1])
    return jsonify({'delayed': bool(pred), 'confidence': prob})

if __name__ == '__main__':
    app.run(port=5001)
