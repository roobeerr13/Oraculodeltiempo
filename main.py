"""Main entrypoint for power consumption forecasting application.

This script can:
1. Run the preprocessing pipeline
2. Start the web application for predictions
"""
from __future__ import annotations
import sys
import os
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
from flask import Flask, render_template, request
from scripts.preprocess_power_consumption import main as preprocess_main

# Initialize Flask app
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

def main() -> int:
    parser = argparse.ArgumentParser(description="Power consumption forecasting pipeline")
    parser.add_argument('--preprocess-only', action='store_true',
                      help='Only run preprocessing, skip web application')
    parser.add_argument('--cli', action='store_true',
                      help='Run in command line mode (no web interface)')
    args = parser.parse_args()
    
    try:
        # Always run preprocessing if data doesn't exist
        if args.preprocess_only or not os.path.exists(os.path.join('outputs', 'X_test.npy')):
            print("Running preprocessing pipeline...")
            preprocess_main()
            if args.preprocess_only:
                return 0
        
        if args.cli:
            # Run in CLI mode
            print("Making prediction in CLI mode...")
            model = load_model()
            current_sequence = prepare_latest_data()
            prediction = model.predict(current_sequence)
            print(f"\nPredicted next value: {prediction[0][0]:.2f}")
        else:
            # Start web application
            print("Starting web application...")
            app.run(debug=True)
        
        return 0
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
