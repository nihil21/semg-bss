from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


def plot_signal(
        s: np.ndarray,
        fs: float = 1,
        n_cols: int = 1,
        fig_size: Optional[Tuple[int, int]] = None
):
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s: np.ndarray
        Signal with shape (n_channels, n_samples).
    fs: float, default=1
        Sampling frequency of the signal.
    n_cols: int, default=1
        Number of columns in the plot.
    fig_size: Optional[Tuple[int, int]], default=None
        Size of Matplotlib figure.
    """

    if fig_size:
        plt.figure(figsize=fig_size)
    n_channels, n_samples = s.shape
    x = np.arange(n_samples) / fs

    # Compute n. of rows
    mod = n_channels % n_cols
    n_rows = n_channels // n_cols if mod == 0 else n_channels // n_cols + mod

    for i in range(n_channels):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f"Channel {i + 1}")
        plt.plot(x, s[i], "tab:blue")
        plt.grid()

    plt.tight_layout()
    plt.show()


def plot_signals(
        s1: np.ndarray,
        s2: np.ndarray,
        fs: float = 1,
        n_cols: int = 1,
        fig_size: Optional[Tuple[int, int]] = None
):
    """Plot side by side multiple signals with multiple channels.

    Parameters
    ----------
    s1: np.ndarray
        First signals with shape (n_channels, n_samples).
    s2: np.ndarray
        Second signals with shape (n_channels, n_samples).
    fs: float, default=1
        Sampling frequency of the two signals.
    n_cols: int, default=1
        Number of columns in the plot.
    fig_size: Optional[Tuple[int, int]], default=None
        Size of Matplotlib figure.
    """
    assert s1.shape[0] == s2.shape[0], "The two signals must have the same number of channels."

    if fig_size:
        plt.figure(figsize=fig_size)
    n_channels, n_samples1 = s1.shape
    _, n_samples2 = s2.shape
    x1 = np.arange(n_samples1) / fs
    x2 = np.arange(n_samples2) / fs

    # Compute n. of rows for one signal
    mod = n_channels % n_cols
    n_rows = n_channels // n_cols if mod == 0 else n_channels // n_cols + mod
    offset = 0
    for i in range(n_rows):
        if i == 0:  # case in which there's only one row
            term = n_cols
        elif i == n_rows - 1:
            term = mod
        else:
            term = n_cols
        for j in range(term):
            idx1 = 2 * offset + j + 1
            plt.subplot(2 * n_rows, n_cols, idx1)
            plt.title(f"Signal 1, channel {offset + j + 1}")
            plt.plot(x1, s1[offset + j], "tab:blue")
            plt.grid()
            idx2 = 2 * offset + n_cols + j + 1
            plt.subplot(2 * n_rows, n_cols, idx2)
            plt.title(f"Signal 2, channel {offset + j + 1}")
            plt.plot(x2, s2[offset + j], "tab:orange")
            plt.grid()
        offset += n_cols

    plt.tight_layout()
    plt.show()
