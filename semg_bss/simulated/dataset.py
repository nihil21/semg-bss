import os
from typing import Tuple

import numpy as np
import scipy.io as sio


def load_semg(
        root: str,
        subject: int,
        mvc: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load data from the 1DoF subset.

    Parameters
    ----------
    root: str
        Path to Hyser dataset root folder.
    subject: int
        Subject id.
    mvc: int
        MVC value (either 10, 30 or 50).

    Returns
    -------
    emg: np.ndarray
        Array containing the simulated sEMG signal.
    gt_spikes: np.ndarray
        Array containing the ground truth spike trains.
    fs_emg: int
        Sampling frequency of the sEMG signal.
    """

    assert mvc in [10, 30, 50], \
        "The MVC value type must be either 10, 30 or 50."

    path = os.path.join(root, f"S{subject + 1:d}_{mvc}MVC.mat")
    data = sio.loadmat(path)
    # Load sEMG data
    emg_tmp = data["sig_out"]
    n_channels = emg_tmp.shape[0] * emg_tmp.shape[1]
    n_samples = emg_tmp[0, 0].shape[1]
    emg = np.zeros(shape=(n_channels, n_samples), dtype=float)
    k = 0
    for i in range(emg_tmp.shape[0]):
        for j in range(emg_tmp.shape[1]):
            emg[k] = emg_tmp[i, j]
            k += 1
    # Load ground truth spike trains
    n_mu = data["sFirings"].shape[1]
    gt_spikes = np.zeros(shape=(n_mu, n_samples), dtype=int)
    for i in range(n_mu):
        spike_loc = data["sFirings"][0, i]
        spike_loc = spike_loc[spike_loc < n_samples]
        gt_spikes[i, spike_loc] = 1
    # Load sampling frequency
    fs_emg = data["fsamp"].item()

    return emg, gt_spikes, fs_emg
