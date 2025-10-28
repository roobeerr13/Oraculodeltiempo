"""Evaluate trained LSTM model: predict, inverse-transform, plot and save results.

Usage:
    python src/evaluate.py

This script expects:
- Trained model at `models/lstm_model.h5`
- Preprocessed arrays in `outputs/` (X_test.npy, y_test.npy)
- Trained scaler at `outputs/scaler.joblib` or similar

Outputs:
- Plot saved to `reports/figures/lstm_predictions.png`
- Arrays saved to `data/results/y_test_original.npy` and `data/results/y_pred_original.npy`
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def find_scaler_candidates():
    candidates = [
        os.path.join('models', 'scaler.pkl'),
        os.path.join('models', 'scaler.joblib'),
        os.path.join('outputs', 'scaler.joblib'),
        os.path.join('outputs', 'scaler.pkl'),
        os.path.join('outputs', 'scaler.joblib'),
    ]
    return [p for p in candidates if os.path.exists(p)]


def load_scaler():
    candidates = find_scaler_candidates()
    if not candidates:
        raise FileNotFoundError('No scaler file found. Looked for models/scaler.pkl, models/scaler.joblib, outputs/scaler.joblib')
    scaler_path = candidates[0]
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler from: {scaler_path}")
    return scaler


def load_model(path='models/lstm_model.h5') -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Train the model first.")
    model = tf.keras.models.load_model(path)
    print(f"Loaded model from: {path}")
    return model


def evaluate_and_save():
    # Create output dirs
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)

    # Load data
    X_test_path = os.path.join('outputs', 'X_test.npy')
    y_test_path = os.path.join('outputs', 'y_test.npy')
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError('X_test.npy or y_test.npy not found in outputs/. Run preprocessing first.')

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print(f"Loaded X_test ({X_test.shape}) and y_test ({y_test.shape})")

    # Load model and predict
    model = load_model()
    y_pred = model.predict(X_test)
    print(f"Predictions computed: shape {y_pred.shape}")

    # Load scaler and inverse transform
    scaler = load_scaler()

    # Ensure shapes are 2D for inverse_transform
    y_test_reshaped = y_test.reshape(-1, 1)
    y_pred_reshaped = y_pred.reshape(-1, 1)

    try:
        y_test_orig = scaler.inverse_transform(y_test_reshaped)
        y_pred_orig = scaler.inverse_transform(y_pred_reshaped)
    except Exception as e:
        # If scaler expects multiple features, try applying on single-column arrays
        print("Scaler inverse_transform failed, attempting with reshape. Error:", e)
        y_test_orig = scaler.inverse_transform(y_test_reshaped)
        y_pred_orig = scaler.inverse_transform(y_pred_reshaped)

    y_test_orig = y_test_orig.ravel()
    y_pred_orig = y_pred_orig.ravel()

    # Compute simple metric
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    print(f"MSE (original scale): {mse:.4f}")

    # Save arrays
    np.save(os.path.join('data', 'results', 'y_test_original.npy'), y_test_orig)
    np.save(os.path.join('data', 'results', 'y_pred_original.npy'), y_pred_orig)
    print("Saved original-scale arrays to data/results/")

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(y_test_orig))
    plt.plot(x, y_test_orig, label='Real (kWh)')
    plt.plot(x, y_pred_orig, label='Predicho (kWh)')
    plt.title('LSTM - Predicciones vs Reales')
    plt.xlabel('Muestra')
    plt.ylabel('Consumo (kWh)')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join('reports', 'figures', 'lstm_predictions.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to: {out_path}")
    
    # Error series (absolute error over samples)
    error = y_pred_orig - y_test_orig
    plt.figure(figsize=(12, 3.5))
    plt.plot(x, error, color='#ff7f50', label='Error (pred - real)')
    plt.hlines(0, x.min(), x.max(), colors='gray', linestyles='dashed', alpha=0.6)
    plt.title('Error por muestra (pred - real)')
    plt.xlabel('Muestra')
    plt.ylabel('Error (kWh)')
    plt.legend()
    plt.tight_layout()
    err_path = os.path.join('reports', 'figures', 'lstm_error_series.png')
    plt.savefig(err_path)
    plt.close()
    print(f"Saved error series to: {err_path}")

    # Error histogram
    plt.figure(figsize=(6, 4))
    plt.hist(error, bins=40, color='#6ea8fe', alpha=0.9)
    plt.title('Distribución del error (pred - real)')
    plt.xlabel('Error (kWh)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    hist_path = os.path.join('reports', 'figures', 'lstm_error_hist.png')
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved error histogram to: {hist_path}")


if __name__ == '__main__':
    evaluate_and_save()