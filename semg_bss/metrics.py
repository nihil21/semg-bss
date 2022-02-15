import time
from typing import Optional

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def silhouette(
        emg: np.ndarray,
        fs: float,
        n: int = 4,
        seed: Optional[int] = None,
        verbose: bool = False
) -> np.ndarray:
    """Compute the silhouette score of a sEMG signal after filtering it with a Butterworth filter.

    Parameters
    ----------
    emg: np.ndarray
        Input sEMG signal with shape (n_channels, n_samples).
    fs: float
        Sampling frequency for the sEMG signal.
    n: int, default=4
        Filter's order.
    seed: Optional[int], default=None
        Seed for the Pseudo-Random Number Generator.
    verbose: bool, default=False
        Whether to log information or not.

    Returns
    -------
    sil: np.ndarray
        The silhouette score for each channel.
    """
    wn = 500 / (fs / 2)
    sos = butter(n, wn, "low", output="sos")
    n_mu = emg.shape[0]
    sil = np.zeros(shape=(n_mu,), dtype=float)
    start = time.time()
    # Iterate over MUs
    for i in range(n_mu):

        if verbose:
            print("\r", end="")
            print(f"Channel {i + 1}/{emg.shape[0]}", end="", flush=True)

        # Filter signal
        emg[i, :] = sosfiltfilt(sos, emg[i, :])
        # Find peaks of squared signal
        sig_sq = np.square(emg[i, :])
        peaks, _ = find_peaks(sig_sq)
        # Perform k-means with 2 clusters (high and small peaks)
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=seed)
        idx = kmeans.fit_predict(sig_sq[peaks].reshape(-1, 1))
        # Compute silhouette score
        sil_ch = silhouette_samples(sig_sq[peaks].reshape(-1, 1), idx, metric="sqeuclidean")
        sil[i] = (sil_ch[idx == 0].mean() + sil_ch[idx == 1].mean()) / 2

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Silhouette computed in {elapsed:.2f} s")

    return sil
