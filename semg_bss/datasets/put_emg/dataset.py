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

import glob
import os
from itertools import groupby

import numpy as np
import pandas as pd
from scipy import signal


def load_put_emg(
    root: str,
    subject: int,
    session: int,
    task_type: str,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Load data from the putEMG dataset.

    Parameters
    ----------
    root : str
        Path to putEMG dataset root folder.
    subject : int
        Subject id.
    session : int
        Session id.
    task_type : {"repeats_long", "repeats_short", "sequential"}
        Task type.

    Returns
    -------
    ndarray
        Array containing the sEMG signal for the given subject, session, and task type.
    list of tuple of (str, int, int)
        List containing, for each action block, the label of the action together with the first and the last samples.
    """

    assert task_type in [
        "repeats_long",
        "repeats_short",
        "sequential"
    ], "The signal type must be either \"repeats_long\", \"repeats_short\" or \"sequential\"."

    # Create path pattern
    path = os.path.join(
        root,
        f"emg_gestures-{subject:02d}-{task_type}-*",
    )
    # Sort files corresponding to pattern and take the one for the given session
    hdf5_file = [file for file in sorted(glob.glob(path))][session - 1]

    # Read DataFrame
    df: pd.DataFrame = pd.read_hdf(hdf5_file)
    # Obtain array with signal
    emg = np.zeros(shape=(24, df.shape[0]))
    for i in range(24):
        emg[i] = df[f"EMG_{i + 1}"].to_numpy() * (5 / 2**12) * (1000 / 200)  # mV
    
    # Define gesture dictionary
    gesture_dict = {
        -1: "rest",
        0: "idle",
        1: "fist",
        2: "flexion",
        3: "extension",
        6: "pinch thumb-index",
        7: "pinch thumb-middle",
        8: "pinch thumb-ring",
        9: "pinch thumb-small"
    }

    # Obtain labels
    labels_tmp = [list(group) for _, group in groupby(df["TRAJ_GT"].values.tolist())]
    labels: list[tuple[str, int, int]] = []
    count = 0
    for cur_label in labels_tmp:
        labels.append(
            (gesture_dict[cur_label[0]], count, count + len(cur_label))
        )
        count += len(cur_label)
    
    return emg, labels


def down_sample(
        emg: np.ndarray,
        labels: list[tuple[str, int, int]],
        old_fs: float,
        new_fs: float
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Down-sample EMG signals and labels to a lower frequency.

    Parameters
    ----------
    emg : ndarray
        EMG signal with shape (n_channels, n_samples).
    labels : list of tuple of (str, int, int)
        List containing, for each action block, the label of the action together with the first and the last samples.
    old_fs : int
        Original sampling frequency of the signal.
    new_fs : int
        New sampling frequency.

    Returns
    -------
    ndarray
        Down-sampled EMG signal with shape (n_channels, n_samples).
    list of tuple of (str, int, int)
        List containing, for each action block, the label of the action
        together with the (down-sampled) first and the last samples.
    """
    # Compute down-sampling factor
    down_factor = int(old_fs / new_fs)
    # Resample at lower frequency
    emg_down = signal.decimate(emg, down_factor)
    labels_down = []
    for i in range(len(labels)):
        # Get boundaries for current label
        l, old_from, old_to = labels[i]
        # Convert them to new sampling frequency
        new_from = int(old_from / old_fs * new_fs)
        new_to = int(old_to / old_fs * new_fs)
        # Save them
        labels_down.append((l, new_from, new_to))

    return emg_down, labels_down
