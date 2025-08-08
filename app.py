import os
if not os.path.exists('model/model.pkl'):
    from train_model import train_and_save_model
    train_and_save_model()
# app.py
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# DB model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(100))
    destination = db.Column(db.String(100))
    preferred_time = db.Column(db.String(50))
    crowd_level = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Load model and encoders
model = pickle.load(open("model/model.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))

@app.route('/')
def home():
    # Use unique station names from dataset
    stations = sorted(encoders['data']['Source'].unique().tolist())
    times = sorted(encoders['data']['Preferred_Time'].unique().tolist())
    return render_template('index.html', stations=stations, times=times)

@app.route('/predict', methods=['POST'])
def predict():
    source = request.form['source']
    destination = request.form['destination']
    preferred_time = request.form['time']

    # Encode inputs
    src_enc = encoders['le_source'].transform([source])[0]
    dst_enc = encoders['le_destination'].transform([destination])[0]
    time_enc = encoders['le_time'].transform([preferred_time])[0]

    # Predict crowd level
    prediction = model.predict([[src_enc, dst_enc, time_enc]])[0]
    crowd_label = encoders['le_crowd'].inverse_transform([prediction])[0]

    # Filter from original dataset
    df = encoders['data']
    filtered = df[(df['Source'] == source) &
                 (df['Destination'] == destination) &
                 (df['Preferred_Time'] == preferred_time)]
    if not filtered.empty:
        match = filtered.iloc[0]
        result = {
            'train': match['Best_Train'],
            'departure': match['Departure'],
            'arrival': match['Arrival'],
            'crowd': crowd_label,
            'standing_time': match['Expected_Standing_Time'],
            'seat_available_after': match['Seat_Likely_Available_After'],
            'alternate_train': match['Alternate_Train']
        }
    else:
        # Fallback result if no match found
        result = {
            'train': 'N/A',
            'departure': 'N/A',
            'arrival': 'N/A',
            'crowd': crowd_label,
            'standing_time': 'N/A',
            'seat_available_after': 'N/A',
            'alternate_train': 'N/A'
        }

    # Log to DB
    entry = Prediction(
        source=source,
        destination=destination,
        preferred_time=preferred_time,
        crowd_level=crowd_label
    )
    db.session.add(entry)
    db.session.commit()

    return render_template('result.html', result=result)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
