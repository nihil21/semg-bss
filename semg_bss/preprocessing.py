"""Copyright 2022 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def filter_signal(
    x: np.ndarray,
    fs: float,
    min_freq: float,
    max_freq: float,
    notch_freqs: tuple[float, ...] = (),
    order: int = 5
) -> np.ndarray:
    """Filter signal with a bandpass filter and notch filters.

    Parameters
    ----------
    x : ndarray
        Signal with shape (n_channels, n_samples).
    fs : float
        Sampling frequency of the signal.
    min_freq : float
        Minimum frequency for bandpass filter.
    max_freq : float
        Maximum frequency for bandpass filter.
    notch_freqs : tuple of (float,), default=()
        Tuple of frequencies to attenuate with notch filters (e.g. for powerline noise).
    order : int, default=5
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_channels, n_samples).
    """
    assert (
        max_freq > min_freq
    ), "The maximum frequency should be greater than the minimum frequency."

    # Apply Butterworth filter
    sos = signal.butter(order, (min_freq, max_freq), "bandpass", fs=fs, output="sos")
    x_filt = signal.sosfiltfilt(sos, x)
    # Apply notch filter
    for freq in notch_freqs:
        b, a = signal.iirnotch(freq, 30, fs)
        x_filt = signal.filtfilt(b, a, x_filt)

    return x_filt


def extend_signal(x: np.ndarray, f_e: int = 0) -> np.ndarray:
    """Extend signal with delayed replicas by a given extension factor.

    Parameters
    ----------
    x : ndarray
        Signal with shape (n_channels, n_samples).
    f_e : int, default=0
        Extension factor.

    Returns
    -------
    ndarray
        Extended signal with shape (f_e * n_channels, n_samples).
    """

    n_obs, n_samples = x.shape
    n_obs_ext = n_obs * f_e
    x_ext = np.zeros(shape=(n_obs_ext, n_samples - f_e + 1), dtype=float)
    for i in range(f_e):
        x_ext[i::f_e] = x[:, f_e - i - 1:n_samples - i]

    return x_ext


def center_signal(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center signal.

    Parameters
    ----------
    x : ndarray
        Signal with shape (n_channels, n_samples).

    Returns
    -------
    ndarray
        Centered signal with shape (n_channels, n_samples).
    ndarray
        Mean vector of the signal with shape (n_channels,).
    """

    x_mean = np.mean(x, axis=1, keepdims=True)
    x_center = x - x_mean

    return x_center, x_mean


def whiten_signal(x: np.ndarray, reg_factor: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Whiten signal using ZCA algorithm.

    Parameters
    ----------
    x : ndarray
        Signal with shape (n_channels, n_samples).
    reg_factor : float, default=0.5
        Regularization factor representing the proportion of eigenvalues 
        that are ignored in the computation of the whitening matrix.

    Returns
    -------
    ndarray
        Whitened signal with shape (n_channels, n_samples).
    ndarray
        Whitening matrix.
    """
    assert 0 <= reg_factor < 1, "The regularization factor must be in range [0, 1[."

    # Compute SVD of correlation matrix
    cov_mtx = np.cov(x)
    u, s, vh = np.linalg.svd(cov_mtx)
    # Regularization: keep only the eigenvalues (and the corresponding eigenvectors)
    # that are greater than the mean of the smallest half of the eigenvalues
    n_eig = s.shape[0]
    n_noise = int(reg_factor * n_eig)
    eig_th = s[n_eig - n_noise:].mean() if n_noise != 0 else -np.inf
    idx = s > eig_th
    # Compute whitening matrix
    d = np.diag(1.0 / np.sqrt(s[idx]))
    white_mtx = u[:, idx] @ d @ vh[idx, :]
    x_white = white_mtx @ x

    return x_white, white_mtx
