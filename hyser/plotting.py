from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


def plot_signals(x: np.ndarray, n_cols: int = 1, fig_size: Optional[Tuple[int, int]] = None):
    """Plot a signal with multiple channels.

    Parameters
    ----------
    x: np.ndarray
        Signal with shape (n_channels, n_samples).
    n_cols: int, default=1
        Number of columns in the plot.
    fig_size: Optional[Tuple[int, int]], default=None
        Size of Matplotlib figure.
    """

    if fig_size:
        plt.figure(figsize=fig_size)
    n_channels = x.shape[0]

    mod = n_channels % n_cols
    n_rows = n_channels // n_cols if mod == 0 else n_channels // n_cols + mod

    for i in range(n_channels):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f"Channel {i + 1}")
        plt.plot(x[i], "tab:blue")
        plt.grid()

    plt.tight_layout()
    plt.show()
