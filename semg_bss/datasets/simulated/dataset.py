"""Copyright 2022 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

from semg_bss.preprocessing import filter_signal


def load_simulated(
    root: str,
    mvc: int,
    snr: int | None = None
) -> list[tuple[np.ndarray, pd.DataFrame, float]]:
    """Load data from the simulated dataset given the MVC value.

    Parameters
    ----------
    root: str
        Path to simulated dataset root folder.
    mvc: int
        MVC value (either 10, 30 or 50).
    snr: int | None, default=None
        Amount of noise in the bandwidth of 20-500 Hz to add to the signal.

    Returns
    -------
    data: list[tuple[np.ndarray, pd.DataFrame, float]]
        List containing the sEMG signal, the ground truth spike trains and the sampling frequency for each simulation.
    """
    assert mvc in [10, 30, 50], "The MVC value type must be either 10, 30 or 50."

    data = []
    for path in sorted(glob.glob(os.path.join(root, f"S*_{mvc}MVC.mat"))):
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

        # Apply noise, if specified
        if snr is not None:
            # Compute signal power and convert to dB
            emg_avg_power = np.mean(np.square(emg), axis=1)
            emg_avg_db = 10 * np.log10(emg_avg_power)
            # Compute noise power
            noise_avg_db = emg_avg_db - snr
            noise_avg_power = 10 ** (noise_avg_db / 10)

            # Generate band-limited noise with given power
            noise = np.zeros_like(emg)
            for i in range(n_channels):
                noise[i] = np.random.standard_normal(n_samples)
                noise[i] = filter_signal(noise[i], fs_emg, min_freq=20, max_freq=500, order=8)
                scale = np.sqrt(noise_avg_power[i]) / noise[i].std()
                noise[i] *= scale

            # Generate white noise with given power
            # noise = np.random.normal(scale=np.sqrt(noise_avg_power), size=(n_channels, n_samples))

            # Noise up the original signal
            emg = emg + noise

        data.append((emg, gt_firings, fs_emg))

    return data
