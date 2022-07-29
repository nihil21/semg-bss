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

import os

import numpy as np
import scipy.io as sio


def load_biowolf(
    root: str,
    n_sig: int
) -> np.ndarray:
    """Load data from the biowolf dataset.

    Parameters
    ----------
    root: str
        Path to biowolf dataset root folder.
    n_sig: int
        Signal id.

    Returns
    -------
    emg: np.ndarray
        Array containing the sEMG signal for the given signal id.
    """
    assert n_sig in (1, 2), "Only two signals are available."

    # Load .mat file
    data = sio.loadmat(os.path.join(root, "dataset4kdry"))
    data = data["ExGData"][0, 0] if n_sig == 1 else data["ExGData2"][0, 0]
    gain = data["SignalGain"].item()
    emg = data["Data"].T * (5 / 2**24) * (1000 / gain)

    return emg
