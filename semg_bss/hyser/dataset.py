import os
import time
from math import ceil

import numpy as np
import wfdb


def load_1dof(
        root: str,
        subject: int,
        session: int,
        sig_type: str = "raw",
        verbose: bool = False,
) -> np.ndarray:
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
    data: np.ndarray
        Array containing sEMG signal for each finger and trial.
    """

    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(root, "1dof_dataset", f"subject{subject + 1:02d}_session{session + 1}")
    data = []
    start = time.time()
    for i in range(1, 6):  # tasks

        if verbose:
            print("\r", end="")
            print(f"Loading task {i}/5", end="", flush=True)

        data_cur = []
        for j in range(1, 4):  # trials
            signal, _ = wfdb.rdsamp(os.path.join(path, f"1dof_{sig_type}_finger{i}_sample{j}"))
            data_cur.append(signal.T)

        data.append(np.stack(data_cur))

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Data loaded in {elapsed:.2f} s")

    return np.stack(data)


def load_mvc(
        root: str,
        subject: int,
        session: int,
        sig_type: str = "raw",
        verbose: bool = False,
) -> np.ndarray:
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
    data: np.ndarray
        Dictionary containing the sEMG signal for each finger.
    """

    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(root, "mvc_dataset", f"subject{subject + 1:02d}_session{session + 1}")
    data = []
    start = time.time()
    for i in range(1, 11):
        direction = "extension" if i % 2 == 0 else "flexion"

        if verbose:
            print("\r", end="")
            print(f"Loading sample {i}/10", end="", flush=True)

        signal, _ = wfdb.rdsamp(os.path.join(path, f"mvc_{sig_type}_finger{ceil(i / 2)}_{direction}"))
        data.append(signal.T)

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Data loaded in {elapsed:.2f} s")

    return np.stack(data)


def load_ndof(
        root: str,
        subject: int,
        session: int,
        sig_type: str = "raw",
        verbose: bool = False,
) -> np.ndarray:
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
    data: np.ndarray
        Array containing sEMG signal for each finger and trial.
    """

    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(root, "ndof_dataset", f"subject{subject + 1:02d}_session{session + 1}")
    data = []
    start = time.time()
    for i in range(1, 16):  # tasks

        if verbose:
            print("\r", end="")
            print(f"Loading task {i}/15", end="", flush=True)

        data_cur = []
        for j in range(1, 3):  # trials
            signal, _ = wfdb.rdsamp(os.path.join(path, f"ndof_{sig_type}_combination{i}_sample{j}"))
            data_cur.append(signal.T)

        data.append(np.stack(data_cur))

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"Data loaded in {elapsed:.2f} s")

    return np.stack(data)
