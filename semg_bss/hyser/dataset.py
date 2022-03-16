import os
import time
from math import ceil

import numpy as np
import wfdb


def load_pr(
        root: str,
        gesture: int,
        subject: int,
        session: int,
        task_type: str,
        sig_type: str,
        task: int,
        trial: int
) -> np.ndarray:
    """Load data from the 1DoF subset.

    Parameters
    ----------
    root: str
        Path to Hyser dataset root folder.
    gesture: int,
        Gesture id.
    subject: int
        Subject id.
    session: int
        Session id.
    task_type: str
        Task type ("maintenance" or "dynamic").
    sig_type: str
        Signal type ("raw", "preprocess" or "force").
    task: int
        Task id.
    trial: int
        Trial id.

    Returns
    -------
    data: np.ndarray
        Array containing sEMG signal for the given gesture, subject, session, trial and task<.
    """

    assert task_type in ["maintenance", "dynamic"], \
        "The signal type must be either \"maintenance\" or \"dynamic\"."
    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(
        root,
        "pr_dataset",
        f"{gesture:02d}",
        f"subject{subject:02d}_session{session}_{task_type}_{sig_type}_trial{trial}_task{task}")
    data, _ = wfdb.rdsamp(path)

    return data.T


def load_1dof(
        root: str,
        subject: int,
        session: int,
        task: int,
        trial: int,
        sig_type: str = "raw"
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
    task: int
        Task id.
    trial: int
        Trial id.
    sig_type: str, default="raw"
        Signal type ("raw", "preprocess" or "force").

    Returns
    -------
    data: np.ndarray
        Array containing sEMG signal for each finger and trial.
    """

    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(
        root,
        "1dof_dataset",
        f"subject{subject:02d}_session{session}",
        f"1dof_{sig_type}_finger{task}_sample{trial}"
    )
    data, _ = wfdb.rdsamp(path)

    return data.T


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
        combination: int,
        trial: int,
        sig_type: str = "raw"
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
    combination: int
        Combination id.
    trial: int
        Trial id.
    sig_type: str, default="raw"
        Signal type ("raw", "preprocess" or "force").

    Returns
    -------
    data: np.ndarray
        Array containing sEMG signal for each finger and trial.
    """

    assert sig_type in ["raw", "preprocess", "force"], \
        "The signal type must be either \"raw\", \"preprocess\" or \"force\"."

    path = os.path.join(
        root,
        "ndof_dataset",
        f"subject{subject:02d}_session{session}",
        f"ndof_{sig_type}_combination{combination}_sample{trial}"
    )
    data, _ = wfdb.rdsamp(path)

    return data.T
