from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='template', static_folder='static')

# ── Model Load ─────────────────────────────────────
with open('model/disaster_model.pkl', 'rb') as f:
    save_data = pickle.load(f)

model    = save_data['model']
FEATURES = save_data['features']
CLASSES  = list(model.classes_)
print("✅ Model loaded successfully!")
print(f"   Classes: {CLASSES}")

# ── Season helper ──────────────────────────────────
def get_season(month):
    month = int(month)
    if month in [6, 7, 8, 9]:    return 1
    elif month in [10, 11, 12]:  return 2
    elif month in [3, 4, 5]:     return 3
    else:                         return 4

# ── Routes ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        year     = float(data.get('year',     2024))
        month    = float(data.get('month',    6))
        deaths   = float(data.get('deaths',   100))
        affected = float(data.get('affected', 50000))

        season       = get_season(month)
        log_deaths   = np.log1p(deaths)
        log_affected = np.log1p(affected)

        input_df = pd.DataFrame(
            [[year, month, season, log_deaths, log_affected]],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        confidence = round(max(proba) * 100, 1)
        prob_dict  = {cls: round(p * 100, 1) for cls, p in zip(CLASSES, proba)}

        return jsonify({
            'disaster':      prediction,
            'confidence':    confidence,
            'probabilities': prob_dict,
            'status':        'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True)