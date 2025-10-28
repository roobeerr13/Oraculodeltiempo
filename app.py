"""Flask web application for power consumption forecasting."""
from __future__ import annotations
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

def load_model():
    """Load the trained LSTM model."""
    model_path = os.path.join('models', 'lstm_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run training.py first to train the model."
        )
    return tf.keras.models.load_model(model_path)

def prepare_latest_data():
    """Get the latest data window for prediction."""
    # Load the last sequence from test data
    # TODO: Update this to load actual latest data in production
    X_test = np.load(os.path.join('outputs', 'X_test.npy'))
    return X_test[-1:]  # Take last sequence as input

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make a new prediction and display it."""
    try:
        # Load model and make prediction
        model = load_model()
        current_sequence = prepare_latest_data()
        prediction = model.predict(current_sequence)
        
        return render_template('index.html',
                             prediction=float(prediction[0][0]),
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    except Exception as e:
        return render_template('index.html', error=str(e))

def main():
    """Run the Flask application."""
    # Ensure required directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Check if model and data exist
    if not os.path.exists(os.path.join('models', 'lstm_model.h5')):
        print("WARNING: Model file not found. Please run training.py first.")
    
    if not os.path.exists(os.path.join('outputs', 'X_test.npy')):
        print("WARNING: Test data not found. Please run preprocessing pipeline first.")
    
    # Run the app
    app.run(debug=True)

if __name__ == '__main__':
    main()