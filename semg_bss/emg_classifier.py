from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf


def df_to_dense(df: pd.DataFrame, n_mu: int, sig_len: float, fs: float) -> np.ndarray:
    """Convert a DataFrame of MUAPTs into an array of ones and zeros (spike/not spike).
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the firing times of every MU.
    n_mu: int
        Number of total MUs.
    sig_len: float
        Length of the signal (in seconds).
    fs: float
        Sampling frequency.
    
    Returns
    -------
    spikes: np.ndarray
        Array of spikes with shape (n_mu, sig_len * fs)
    """
    spikes = np.zeros(shape=(n_mu, int(sig_len * fs)), dtype=int)
    for mu in range(n_mu):
        spikes_idx = (df[df["MU index"] == mu]["Firing time"].to_numpy() * fs).astype(int)
        spikes[mu, spikes_idx] = 1
    
    return spikes


def build_mlp_classifier(
    n_in: int,
    n_out: int,
    hidden_struct: tuple[int, ...],
    optimizer: str | tf.keras.optimizers.Optimizer,
    regularize: bool = False
) -> tf.keras.Model:
    """Build and compile a MLP for classification.
    
    Parameters
    ----------
    n_in: int
        Number of input units.
    n_out: int
        Number of output units.
    hidden_struct: tuple[int, ...]
        Tuple containing, for each hidden layer, the number of units.
    optimizer: str | tf.keras.optimizers.Optimizer
        Optimizer.
    regularize: bool, default=False
        Whether to add L2 regularization (1e-4) to the hidden layers.
    """
    # Input layer
    in_layer = tf.keras.Input(shape=(n_in,))
    # Hidden layers
    x = in_layer
    for n_cur_hid in hidden_struct:
        if regularize:
            x = tf.keras.layers.Dense(
                n_cur_hid,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(x)
        else:
            x = tf.keras.layers.Dense(n_cur_hid, activation="relu")(x)
    # Output layer
    out_activation = "sigmoid" if n_out == 1 else "softmax" 
    out_layer = tf.keras.layers.Dense(n_out, activation=out_activation)(x)
    
    # Create Model and compile it
    model = tf.keras.Model(inputs=in_layer, outputs=out_layer)
    loss = "binary_crossentropy" if n_out == 1 else "sparse_categorical_crossentropy"
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    return model
