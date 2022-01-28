import time
from typing import Tuple

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def spike_detection(emg: np.ndarray, threshold: float, verbose: bool = False) -> np.ndarray:
    """Find spikes via peak detection and k-means clustering.

    Parameters
    ----------
    emg: np.ndarray
        Input sEMG signal with shape (n_channels, n_samples).
    threshold: float
        Threshold on the silhouette score for considering a spike train valid.
    verbose: bool, default=False
        Whether to log information or not.
    """
    spike_train = np.zeros(shape=emg.shape, dtype=float)
    k = 0  # keep track of valid spikes
    start = time.time()
    for i in range(emg.shape[0]):

        if verbose:
            print("\r", end="")
            print(f"Channel {i + 1}/{emg.shape[0]}", end="", flush=True)

        # Find peaks of squared signal
        sig_sq = np.square(emg[i, :])
        peaks, _ = find_peaks(sig_sq)
        # Perform k-means with 2 clusters (high and small peaks)
        kmeans = KMeans(n_clusters=2, init="k-means++")
        kmeans.fit(sig_sq[peaks].reshape(-1, 1))
        idx = kmeans.labels_
        c = kmeans.cluster_centers_
        # Consider only high peaks (i.e. cluster with highest centroid)
        high_cluster_idx = np.argmax(c)
        spike_loc = peaks[idx == high_cluster_idx]
        # Compute SIL
        sil = silhouette_score(sig_sq[peaks].reshape(-1, 1), idx)

        if sil > threshold:
            # Create spike train
            spike_train[k, spike_loc] = 1
            k = k + 1
    spike_train = spike_train[:k, :]

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Spike detected in {elapsed:.2f} s")

    return spike_train


def replicas_removal(
        spike_train: np.ndarray,
        emg: np.ndarray,
        fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean the detected spike trains by removing replicas.

    Parameters
    ----------
    spike_train: np.ndarray
        Input spike trains with shape (n_channels, n_samples).
    emg: np.ndarray
        Input sEMG signal with shape (n_channels, n_samples).
    fs: float
        Sampling frequency for the sEMG signal.

    Returns
    -------
    spike_train_good: np.ndarray
        The cleaned spike trains with shape (n_channels, n_samples).
    emg_good: np.ndarray
        The cleaned sEMG signal with shape (n_channels, n_samples).
    """

    time_steps_tmp = np.arange(start=1. / fs, stop=spike_train.shape[1] / fs, step=1. / fs)
    rec_len = time_steps_tmp[-1]

    # Step 1
    firings = np.sum(spike_train, axis=1)
    valid_index_tmp = np.nonzero((firings > 4 * rec_len) & (firings < 35 * rec_len))[0]
    n_valid = len(valid_index_tmp)

    # Step 2
    time_steps = time_steps_tmp.reshape(-1, 1) * np.ones(shape=(1, n_valid), dtype=float)
    for i in range(n_valid):  # iterate over valid channels
        loc = np.nonzero(spike_train[valid_index_tmp[i], :] == 1)[0]  # get idx of spikes in current channel
        diff_loc = np.diff(loc)  # compute relative distance between spikes
        loc_mask = diff_loc < fs * 0.02  # create a boolean mask for distance based on threshold
        for j in range(len(loc_mask)):  # iterate over array of relative distances
            # Merge spikes that are too near to each other
            if loc_mask(j):
                peak1 = emg[i, loc[j]]
                peak2 = emg[i, loc[j + 1]]
                if peak1 >= peak2:
                    spike_train[valid_index_tmp[i], loc[j + 1]] = 0
                else:
                    spike_train[valid_index_tmp[i], loc[j]] = 0

        raise NotImplementedError("Not implemented yet.")
