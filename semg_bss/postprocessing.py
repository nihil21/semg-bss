import time
from math import floor
from typing import Optional

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def spike_detection(
        emg: np.ndarray,
        threshold: float = 0.6,
        seed: Optional[int] = None,
        verbose: bool = False
) -> np.ndarray:
    """Find spikes via peak detection and k-means clustering.

    Parameters
    ----------
    emg: np.ndarray
        Input sEMG signal with shape (n_channels, n_samples).
    threshold: float, default=0.6
        Threshold on the silhouette index for considering a spike valid.
    seed: Optional[int], default=None
        Seed for the Pseudo-Random Number Generator.
    verbose: bool, default=False
        Whether to log information or not.

    Returns
    -------
    spike_train: np.ndarray
        The spike train associated to the sEMG signal.
    """

    spike_train = np.zeros(shape=emg.shape, dtype=float)
    start = time.time()
    for i in range(emg.shape[0]):

        if verbose:
            print("\r", end="")
            print(f"Channel {i + 1}/{emg.shape[0]}", end="", flush=True)

        # Find peaks of squared signal
        sig_sq = np.square(emg[i, :])
        peaks, _ = find_peaks(sig_sq)
        # Perform k-means with 2 clusters (high and small peaks)
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=seed)
        kmeans.fit(sig_sq[peaks].reshape(-1, 1))
        idx = kmeans.labels_
        c = kmeans.cluster_centers_
        # Compute silhouette
        # sil_ch = silhouette_samples(sig_sq[peaks].reshape(-1, 1), idx, metric="sqeuclidean")
        # Consider only high peaks (i.e. cluster with highest centroid)
        high_cluster_idx = np.argmax(c)
        spike_loc = peaks[(idx == high_cluster_idx)]  # ] & (sil_ch > threshold)]
        # Create spike train
        spike_train[i, spike_loc] = 1

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Spike detected in {elapsed:.2f} s")

    return spike_train


def _sync_correlation(
        fire_events_ref: np.ndarray,
        fire_events_sec: np.ndarray,
        win_len: float,
        n_bin: int
) -> bool:
    """Clean the detected spike trains by removing replicas.

    Parameters
    ----------
    fire_events_ref: np.ndarray
        Time of firing events for reference MU.
    fire_events_sec: np.ndarray
        Time of firing events for secondary MU.
    win_len: float
        Length of the window for sync calculation.
    n_bin: int
        Number of bins.

    Returns
    -------
    in_sync: bool
        Whether the two spike trains are in sync or not.
    """
    syn = []
    for fire_event_ref in fire_events_ref:
        fire_interval = fire_events_sec - fire_event_ref
        idx = np.nonzero((fire_interval > -win_len) & (fire_interval < win_len))[0]
        if idx.size != 0:
            syn.append(fire_interval[idx[0]])

    # Compute histogram of relative timing
    # syn = np.array(syn)
    # syn = np.where(
    #     syn == syn.min(initial=0),
    #     syn + 1e-16,
    #     syn
    # )  # make hist compatible with MATLAB implementation
    h, x_bin = np.histogram(syn, n_bin)
    x_bin = x_bin[:-1] + np.diff(x_bin) / 2  # compute bin center
    peak_center = x_bin[h == h.max(initial=0)]
    idx = np.nonzero((syn > peak_center[0] - 0.001) & (syn < peak_center[0] + 0.001))[0]
    common = idx.shape[0] / fire_events_ref.shape[0]
    in_sync = common > 0.5

    return in_sync


def replicas_removal(
        spike_train: np.ndarray,
        emg: np.ndarray,
        fs: float
) -> np.ndarray:
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
    valid_index: np.ndarray
        The cleaned sEMG signal with shape (n_channels, n_samples).
    """

    time_steps = np.arange(start=1. / fs, stop=(spike_train.shape[1] + 1) / fs, step=1. / fs)
    rec_len = time_steps[-1]

    # Step 1
    n_firings = np.sum(spike_train, axis=1)
    valid_index_tmp = np.nonzero((n_firings > 4 * rec_len) & (n_firings < 35 * rec_len))[0]
    n_valid = len(valid_index_tmp)

    # Step 2
    fire_events = []
    rep = 3  # repeat 3 times
    for i in range(rep):
        for j in range(n_valid):  # iterate over valid channels
            loc = np.nonzero(spike_train[valid_index_tmp[j]] == 1)[0]  # get idx of spikes in current channel
            diff_loc = np.diff(loc)  # compute relative distance between spikes
            loc_mask = diff_loc < fs * 0.02  # create a boolean mask for distance based on threshold
            for k in range(len(loc_mask)):  # iterate over array of relative distances
                # Merge spikes that are too near to each other
                if loc_mask[k]:
                    peak1 = emg[j, loc[k]]
                    peak2 = emg[j, loc[k + 1]]
                    if peak1 >= peak2:
                        spike_train[valid_index_tmp[j], loc[k + 1]] = 0
                    else:
                        spike_train[valid_index_tmp[j], loc[k]] = 0

            if i == rep - 1:
                fire_events.append(time_steps[spike_train[valid_index_tmp[j]] == 1])

    # Step 3
    n_mu = len(fire_events)
    count = 0
    index = [i for i in range(n_mu)]

    while len(index) != count:
        index_removal = []
        # Find index of replicas by checking synchronization
        for i in range(1, len(index) - count):
            in_sync = _sync_correlation(fire_events[count], fire_events[count + i], 0.01, 10)
            if in_sync:
                index_removal.append(count + i)
        # Remove indexes
        for idx in reversed(index_removal):
            del fire_events[idx]
            del index[idx]
        count += 1

    valid_index = valid_index_tmp[index]

    return valid_index
