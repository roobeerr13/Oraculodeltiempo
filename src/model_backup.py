"""backup one model file with training and saving functionality."""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def build_model(input_shape: tuple) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_save(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 100, batch_size: int = 32, model_path: str | None = None):
    # Determine model path and ensure directory exists
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.h5')
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    print("Model summary:")
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=2
    )

    # Save training loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, 'training_loss.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training loss plot to: {plot_path}")

    # Save model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.h5')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Saved trained model to: {model_path}")
    return model, history


def load_data(outputs_dir: str | None = None):
    if outputs_dir is None:
        outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    X_train = np.load(os.path.join(outputs_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(outputs_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(outputs_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(outputs_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()
    # Train and save model (default 100 epochs)
    train_and_save(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)


if __name__ == '__main__':
    main()