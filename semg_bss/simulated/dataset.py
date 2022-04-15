from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio


def load_simulated(
    root: str,
    mvc: int,
) -> list[tuple[np.ndarray, pd.DataFrame, float]]:
    """Load data from the 1DoF subset.

    Parameters
    ----------
    root: str
        Path to Hyser dataset root folder.
    mvc: int
        MVC value (either 10, 30 or 50).

    Returns
    -------
    data: list[tuple[np.ndarray, pd.DataFrame, float]]
        List containing the sEMG signal, the ground truth spike trains and the sampling frequency for each simulation.
    """
    assert mvc in [10, 30, 50], "The MVC value type must be either 10, 30 or 50."

    data = []
    for path in glob.glob(os.path.join(root, f"S*_{mvc}MVC.mat")):
        cur_data = sio.loadmat(path)

        # Load sEMG data
        emg_tmp = cur_data["sig_out"]
        n_channels = emg_tmp.shape[0] * emg_tmp.shape[1]
        n_samples = emg_tmp[0, 0].shape[1]
        emg = np.zeros(shape=(n_channels, n_samples), dtype=float)
        k = 0
        for i in range(emg_tmp.shape[0]):
            for j in range(emg_tmp.shape[1]):
                emg[k] = emg_tmp[i, j]
                k += 1

        # Load sampling frequency
        fs_emg = cur_data["fsamp"].item()

        # Load ground truth spike trains
        n_mu = cur_data["sFirings"].shape[1]
        firings_tmp: list[dict[str, float]] = []
        for i in range(n_mu):
            spike_loc = cur_data["sFirings"][0, i]
            spike_loc = spike_loc[spike_loc < n_samples]

            f_rate = spike_loc.size / n_samples * fs_emg
            firings_tmp.extend(
                [
                    {"MU index": i, "Firing time": s / fs_emg, "Firing rate": f_rate}
                    for s in spike_loc
                ]
            )
        # Convert to Pandas DataFrame
        gt_firings = pd.DataFrame(firings_tmp)

        data.append((emg, gt_firings, fs_emg))

    return data
