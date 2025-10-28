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
from flask import Flask, render_template, request, send_file
from sklearn.metrics import mean_squared_error


def get_model_description():
    """Return a short human-readable description of the loaded model."""
    try:
        model = load_model()
    except Exception:
        return None
    parts = []
    for layer in model.layers:
        name = layer.__class__.__name__
        cfg = ''
        # try to extract units if present
        if hasattr(layer, 'units'):
            cfg = f"({getattr(layer, 'units')})"
        parts.append(f"{name}{cfg}")
    return ' → '.join(parts)


def compute_evaluation_metric():
    """Compute MSE from saved results if available (data/results/*.npy).
    Returns float or None.
    """
    y_test_path = os.path.join('data', 'results', 'y_test_original.npy')
    y_pred_path = os.path.join('data', 'results', 'y_pred_original.npy')
    if os.path.exists(y_test_path) and os.path.exists(y_pred_path):
        try:
            y_test = np.load(y_test_path)
            y_pred = np.load(y_pred_path)
            return float(mean_squared_error(y_test, y_pred))
        except Exception:
            return None
    return None
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
    # Pass explicit None values to avoid 'undefined' in the template
    # Also indicate whether an evaluation plot is available
    plot_path = os.path.join('reports', 'figures', 'lstm_predictions.png')
    plot_exists = os.path.exists(plot_path)
    model_description = get_model_description()
    mse = compute_evaluation_metric()
    return render_template('index.html', prediction=None, timestamp=None, plot_exists=plot_exists,
                           model_description=model_description, mse=mse)


@app.route('/report/figure')
def report_figure():
    """Serve the evaluation figure if present."""
    path = os.path.join('reports', 'figures', 'lstm_predictions.png')
    if os.path.exists(path):
        # send_file will set appropriate headers
        return send_file(path, mimetype='image/png')
    else:
        return ("No figure available", 404)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a new prediction and display it."""
    try:
        # If user provided a manual sequence, use it; otherwise use latest test sequence
        manual = request.form.get('manual_sequence', '')
        model = load_model()

        if manual and manual.strip():
            # Parse comma-separated values
            try:
                vals = [float(v.strip()) for v in manual.split(',') if v.strip() != '']
            except ValueError:
                return render_template('index.html', error='Error: la secuencia manual contiene valores no numéricos.', plot_exists=os.path.exists(os.path.join('reports','figures','lstm_predictions.png')))

            # Determine expected timesteps from model input shape
            # model.input_shape is like (None, timesteps, features)
            input_shape = model.input_shape
            if len(input_shape) >= 3 and input_shape[1] is not None:
                timesteps = int(input_shape[1])
            else:
                # Fallback: infer from X_test if available
                try:
                    X_test = np.load(os.path.join('outputs', 'X_test.npy'))
                    timesteps = X_test.shape[1]
                except Exception:
                    return render_template('index.html', error='No se puede determinar la ventana temporal (timesteps).', plot_exists=os.path.exists(os.path.join('reports','figures','lstm_predictions.png')))

            # If user provided a different length, allow padding/truncation:
            if len(vals) < timesteps:
                pad_len = timesteps - len(vals)
                pad_value = vals[-1] if len(vals) > 0 else 0.0
                vals = [pad_value] * pad_len + vals
            elif len(vals) > timesteps:
                # keep the most recent `timesteps` values
                vals = vals[-timesteps:]

            # Build input array: shape (1, timesteps, features)
            arr = np.array(vals, dtype=float).reshape(1, timesteps, 1)
            prediction = model.predict(arr)
        else:
            current_sequence = prepare_latest_data()
            prediction = model.predict(current_sequence)

        model_description = get_model_description()
        mse = compute_evaluation_metric()
        return render_template('index.html',
                               prediction=float(prediction[0][0]),
                               timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               plot_exists=os.path.exists(os.path.join('reports','figures','lstm_predictions.png')),
                               model_description=model_description,
                               mse=mse)
    except Exception as e:
        model_description = get_model_description()
        mse = compute_evaluation_metric()
        return render_template('index.html', error=str(e), plot_exists=os.path.exists(os.path.join('reports','figures','lstm_predictions.png')),
                               model_description=model_description, mse=mse)

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
            # For the web app, we bind to 0.0.0.0 only if needed; default is localhost
            app.run(debug=True)
        
        return 0
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
