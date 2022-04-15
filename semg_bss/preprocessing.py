import numpy as np
from scipy import signal


def filter_signal(
    x: np.ndarray,
    fs: float,
    min_freq: float,
    max_freq: float,
    notch_freqs: list[float],
    order: int = 5
) -> np.ndarray:
    """Filter sEMG signal with a Butterworth bandpass filter.

    Parameters
    ----------
    x: np.ndarray
        sEMG data with shape (n_channels, n_samples).
    fs: float
        Frequency of the sEMG signal.
    min_freq: float
        Minimum frequency.
    max_freq: float
        Maximum frequency.
    order: int, default=4
        Order of the Butterworth filter.

    Returns
    -------
    x_filt: np.ndarray
        Filtered sEMG signal with shape (n_channels, n_samples).
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
    """Extend sEMG signal by a given extension factor.

    Parameters
    ----------
    x: np.ndarray
        sEMG data with shape (n_channels, n_samples).
    f_e: int, default=0
        Extension factor.

    Returns
    -------
    x_ext: np.ndarray
        Extended sEMG signal with shape (f_e * n_channels, n_samples).
    """

    n_obs, n_samples = x.shape
    n_obs_ext = n_obs * f_e
    x_ext = np.zeros(shape=(n_obs_ext, n_samples), dtype=float)
    for i in range(f_e):
        x_ext[i::f_e, i:] = x[:, : n_samples - i]

    return x_ext


def center_signal(x: np.ndarray) -> np.ndarray:
    """Center sEMG signal.

    Parameters
    ----------
    x: np.ndarray
        sEMG data with shape (n_channels, n_samples).

    Returns
    -------
    x_center: np.ndarray
        Centered sEMG signal with shape (n_channels, n_samples).
    """

    x_mean = np.mean(x, axis=1, keepdims=True)
    x_center = x - x_mean

    return x_center


def whiten_signal(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whiten sEMG signal using ZCA algorithm.

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

    # Compute SVD of correlation matrix
    cov_mtx = np.cov(x)
    u, s, vh = np.linalg.svd(cov_mtx)
    # Regularization: keep only the eigenvalues (and the corresponding eigenvectors)
    # that are greater than the mean of the smallest half of the eigenvalues
    reg_factor = 0.5
    n_eig = s.shape[0]
    n_noise = int(reg_factor * n_eig)
    eig_th = s[n_eig - n_noise :].mean()
    idx = s > eig_th
    # Compute whitening matrix
    d = np.diag(1.0 / np.sqrt(s[idx]))
    white_mtx = u[:, idx] @ d @ vh[idx, :]
    x_white = white_mtx @ x

    return x_white, white_mtx
