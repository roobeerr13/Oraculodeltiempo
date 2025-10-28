from __future__ import annotations
import tensorflow as tf


def build_model(input_shape: tuple[int, ...], units: int = 50, learning_rate: float = 1e-3) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=units, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(units=units),
        tf.keras.layers.Dense(1)  # Single target value prediction
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model



