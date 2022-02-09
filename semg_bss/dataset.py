import os
import time
from math import ceil
from typing import Dict

import numpy as np
import wfdb


def load_1dof(
        root: str,
        subject: int,
        session: int,
        sig_type: str = "raw",
        verbose: bool = False,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Load data from the 1DoF subset.

    Parameters
    ----------
    root: str
        Path to Hyser dataset root folder.
    subject: int
        Subject id.
    session: int
        Session id.
    sig_type: str, default="raw"
        Signal type ("raw", "preprocess" or "force").
    verbose: bool, default=False
        Whether to log information or not.

    Returns
    -------
    data: Dict[int, Dict[int, np.ndarray]]
        Dictionary containing the sEMG signal for each finger and sample
    """

    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(root, "1dof_dataset", f"subject{subject:02d}_session{session}")
    data = {}
    start = time.time()
    for i in range(1, 6):

        if verbose:
            print("\r", end="")
            print(f"Loading task {i}/5", end="", flush=True)

        finger_data = {}
        for j in range(1, 4):
            signal, _ = wfdb.rdsamp(os.path.join(path, f"1dof_{sig_type}_finger{i}_sample{j}"))
            finger_data[j] = signal.T

        data[i] = finger_data

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Data loaded in {elapsed:.2f} s")

    return data


def load_mvc(
        root: str,
        subject: int,
        session: int,
        sig_type: str = "raw",
        verbose: bool = False,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Load data from the MVC subset.

    Parameters
    ----------
    root: str
        Path to Hyser dataset root folder.
    subject: int
        Subject id.
    session: int
        Session id.
    sig_type: str, default="raw"
        Signal type ("raw", "preprocess" or "force").
    verbose: bool, default=False
        Whether to log information or not.

    Returns
    -------
    data: Dict[int, Dict[int, np.ndarray]]
        Dictionary containing the sEMG signal for each finger and sample
    """

    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(root, "mvc_dataset", f"subject{subject:02d}_session{session}")
    data = {}
    start = time.time()
    for i in range(1, 11):
        direction = "extension" if i % 2 == 0 else "flexion"

        if verbose:
            print("\r", end="")
            print(f"Loading sample {i}/10", end="", flush=True)

        signal, _ = wfdb.rdsamp(os.path.join(path, f"mvc_{sig_type}_finger{ceil(i / 2)}_{direction}"))
        data[i] = signal.T

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Data loaded in {elapsed:.2f} s")

    return data
