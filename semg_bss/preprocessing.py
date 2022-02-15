from typing import Tuple

import numpy as np


def extend_signal(x: np.ndarray, r: int = 0) -> np.ndarray:
    """Extend sEMG signal by a given extension factor.

    Parameters
    ----------
    x: np.ndarray
        sEMG data with shape (n_channels, n_samples).
    r: int, default=0
        Extension factor.

    Returns
    -------
    x_ext: np.ndarray
        Extended sEMG signal with shape (r * n_channels, n_samples).
    """

    n_obs, n_samples = x.shape
    n_obs_ext = n_obs * (r + 1)
    x_ext = np.zeros(shape=(n_obs_ext, n_samples), dtype=float)
    x_ext[0:n_obs] = x
    if r != 0:
        for i in range(r):
            start_row = n_obs * (i + 1)
            stop_row = n_obs * (i + 2)
            x_ext[start_row:stop_row, i + 1:] = x[:, :-(i + 1)]

    return x_ext


def center_signal(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center sEMG signal.

    Parameters
    ----------
    x: np.ndarray
        sEMG data with shape (n_channels, n_samples).

    Returns
    -------
    x_center: np.ndarray
        Centered sEMG signal with shape (n_channels, n_samples).
    x_mean: np.ndarray
        Mean of the original signal.
    """

    x_mean = np.mean(x, axis=1, keepdims=True)
    x_center = x - x_mean

    return x_center, x_mean


def whiten_signal(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Whiten sEMG signal.

    Parameters
    ----------
    x: np.ndarray
        sEMG data with shape (n_channels, n_samples).

    Returns
    -------
    x_white: np.ndarray
        Whitened sEMG signal with shape (n_channels, n_samples).
    white_mtx: np.ndarray
        Whitening matrix.
    """

    cov_mtx = np.cov(x)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mtx)
    eps = 1e-10
    d = np.diag(1. / (eig_vals + eps)**0.5)
    white_mtx = eig_vecs @ d @ eig_vecs.T
    x_white = white_mtx @ x

    return x_white, white_mtx
