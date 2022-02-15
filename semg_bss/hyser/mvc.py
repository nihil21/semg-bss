from math import floor
from typing import Dict

import numpy as np
import resampy
from scipy.signal import butter, sosfiltfilt

from .dataset import load_mvc


def get_mvc(
        root: str,
        subject: int,
        session: int,
        verbose: bool = False,
) -> np.ndarray:
    """Compute the MVC value for the given subject and session.

    Parameters
    ----------
    root: str
        Path to Hyser dataset root folder.
    subject: int
        Subject id.
    session: int
        Session id.
    verbose: bool, default=False
        Whether to log information or not.

    Returns
    -------
    mvc: np.ndarray
        MVC value for each finger and trial.
    """

    force_data = load_mvc(root, subject, session, "force", verbose)

    mvc = np.zeros(shape=(5, 2), dtype=float)

    for i in range(10):
        finger = i // 2
        direction = i % 2
        force = np.abs(force_data[i, finger])
        force = np.sort(force)[::-1]
        mvc[finger, direction] = np.mean(force[:200])

    return mvc


def normalize_force(
        force: np.ndarray,
        mvc: np.ndarray
) -> np.ndarray:
    """Normalize the force data by the corresponding MVC value.

    Parameters
    ----------
    force: Dict[int, Dict[int, np.ndarray]]
        Force data of each finger, trial and muscle.
    mvc: np.ndarray
        MVC value for each finger and trial.

    Returns
    -------
    force_norm: np.ndarray
        Force data normalized by the MVC.
    """

    n_rows, n_cols, _, _ = force.shape
    force_norm = []
    for i in range(n_rows):
        force_norm_cur = []
        for j in range(n_cols):
            force_cur = force[i, j]
            for u in range(force_cur.shape[0]):
                for v in range(force_cur.shape[1]):
                    if force_cur[u, v] < 0:
                        force_cur[u, v] = force_cur[u, v] / mvc[u, 0]
                    else:
                        force_cur[u, v] = force_cur[u, v] / mvc[u, 1]
            force_norm_cur.append(force_cur)
        force_norm.append(np.stack(force_norm_cur))

    return np.stack(force_norm)


def preprocess_force(
        force: np.ndarray,
        window_size: float,
        step_len: float,
        fs_force: float,
        fs_emg: float
) -> np.ndarray:
    """Preprocess the force data by means of a Butterworth filter, resampling and feature extraction.

    Parameters
    ----------
    force: np.ndarray
        Force data of each finger, trial and muscle.
    window_size: float
        Window size for feature extraction.
    step_len: float
        Step length for feature extraction.
    fs_force: float
        Sampling frequency for the force signal.
    fs_emg: float
        Sampling frequency for the sEMG signal.

    Returns
    -------
    force_norm: np.ndarray
        Force data preprocessed.
    """

    n_rows, n_cols, _, _ = force.shape

    window_sample = floor(window_size * fs_emg)
    step_sample = floor(step_len * fs_emg)

    force_preprocessed = []
    for i in range(n_rows):
        force_preprocessed_cur = []
        for j in range(n_cols):
            force_cur = force[i, j]
            # Filter force signal
            wn = 10 / (fs_force / 2)
            sos = butter(8, wn, "low", output="sos")
            force_cur = sosfiltfilt(sos, force_cur)
            # Resample force signal to sEMG sampling frequency
            force_cur = resampy.resample(force_cur, fs_force, fs_emg)
            n_channels, n_samples = force_cur.shape
            # Extract features from the resampled signal
            idx = 0
            force_preprocessed_tmp = np.zeros(shape=(n_channels, n_samples // step_sample), dtype=float)
            for u in range(0, n_samples - window_sample + 1, step_sample):
                force_preprocessed_tmp[:, idx] = force_cur[:, u:u + window_sample].mean(axis=1)
                idx = idx + 1
            force_preprocessed_cur.append(force_preprocessed_tmp)
        force_preprocessed.append(np.stack(force_preprocessed_cur))

    return np.stack(force_preprocessed)


def estimate_firing_rate(
        spike_train: np.ndarray,
        group_idx: np.ndarray,
        window_size: float,
        step_len: float,
        fs_emg: float,
):
    """Estimate the firing rate from the spike train.

    Parameters
    ----------
    spike_train: np.ndarray
        Input spike train.
    group_idx: np.ndarray
        .
    window_size: float
        Window size for feature extraction.
    step_len: float
        Step length for feature extraction.
    fs_emg: float
        Sampling frequency for the sEMG signal.

    Returns
    -------
    firing_rate: np.ndarray
        Estimated firing rate.
    time_win: np.ndarray
        Time window of firing events.
    """

    n_samples = spike_train.shape[1]
    window_sample = floor(window_size * fs_emg)
    step_sample = floor(step_len * fs_emg)

    firing_rate = []
    for i in range(0, n_samples - window_sample + 1, step_sample):
        firing_rate_cur = []
        for idx in group_idx:
            spike_seg_tmp = spike_train[idx, i:i + window_sample]
            firing_tmp = spike_seg_tmp.sum()
            if firing_tmp != 0:
                firing_rate_cur.append(np.sum(spike_seg_tmp) / window_size)
            else:
                firing_rate_cur.append(0)
        firing_rate.append(np.array(firing_rate_cur))
    firing_rate = np.stack(firing_rate)
    time_max = (n_samples - window_sample / 2 + 1) / fs_emg
    time_win = np.arange(
        start=window_sample / 2 / fs_emg,
        stop=time_max,
        step=step_sample / fs_emg
    )

    return firing_rate.T, time_win
