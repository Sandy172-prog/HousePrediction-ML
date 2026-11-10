from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load column metadata
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']


@app.route('/')
def home():
    with open('columns.json', 'r') as f:
        data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]   # skip sqft, bath, bhk
    locations = sorted(locations)  # optional: sort alphabetically
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    try:
        loc_index = data_columns.index(location)
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    x_df = pd.DataFrame([x], columns=data_columns)
    prediction = model.predict(x_df)[0]

    return render_template(
        'result.html',
        prediction=round(prediction, 2),
        location=location,
        sqft=sqft,
        bath=bath,
        bhk=bhk
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)