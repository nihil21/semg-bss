from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()


def plot_signal(
        s: np.ndarray,
        fs: float = 1,
        n_cols: int = 1,
        fig_size: tuple[int, int] | None = None
) -> None:
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s: np.ndarray
        Signal with shape (n_channels, n_samples).
    fs: float, default=1
        Sampling frequency of the signal.
    n_cols: int, default=1
        Number of columns in the plot.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    n_channels, n_samples = s.shape
    x = np.arange(n_samples) / fs

    # Compute n. of rows
    mod = n_channels % n_cols
    n_rows = n_channels // n_cols if mod == 0 else n_channels // n_cols + mod

    if fig_size is not None:
        plt.figure(figsize=fig_size)

    for i in range(n_channels):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(x, s[i])
        plt.title(f"Channel {i}")

    plt.tight_layout()
    plt.show()


def plot_sub(
        sig_list: list[np.ndarray | pd.DataFrame],
        plot_fn: Callable[..., None],
        title_list: list[str],
        n_cols: int,
        fig_size: tuple[int, int] | None,
        **kwargs: Any
) -> None:
    """Create multiple subplots from a single function and multiple signals.

    Parameters
    ----------
    sig_list: list[np.ndarray | pd.DataFrame]
        List of input signals.
    plot_fn: Callable[[Any], None]
        Plotting function.
    title_list: list[str]
        List of titles for subplots.
    n_cols: int
        Number of columns in the plot.
    fig_size: tuple[int, int] | None
        Height and width of the plot.
    """
    assert len(sig_list) == len(title_list)
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    # Compute n. of rows
    n_plots = len(sig_list)
    mod = n_plots % n_cols
    n_rows = n_plots // n_cols if mod == 0 else n_plots // n_cols + mod

    for i, (sig, title) in enumerate(zip(sig_list, title_list)):
        plt.subplot(n_rows, n_cols, i + 1)
        plot_fn(sig, **kwargs)
        plt.title(title)

    plt.tight_layout()


def plot_correlation(
        s: np.ndarray,
        fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the correlation matrix of the given arrays.

    Parameters
    ----------
    s: np.ndarray
        Input array with shape (n_channels, n_samples).
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    plt.imshow(np.corrcoef(s))
    plt.grid(None)


def raster_plot(
        firings: pd.DataFrame,
        sig_len: float,
        fig_size: tuple[int, int] | None = None
) -> None:
    """Plot a raster plot of the firing activity of a group of neurons.

    Parameters
    ----------
    firings: pd.DataFrame
        A DataFrame with columns "MU index", "Firing time" and "Firing rate" describing the firing activity of neurons.
    sig_len: float
        Length of the signal (in seconds).
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    g = sns.scatterplot(
        data=firings,
        x="Firing time",
        y="MU index",
        hue="Firing rate",
        palette="flare"
    )
    g.set(xlim=(0, sig_len))
    g.set(ylim=(-1, firings["MU index"].max() + 1))

    # Color bar
    norm = plt.Normalize(firings["Firing rate"].min(), firings["Firing rate"].max())
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    sm.set_array([])
    g.get_legend().remove()
    g.figure.colorbar(sm)
