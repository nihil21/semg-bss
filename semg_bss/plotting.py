from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .snn import MUAPTClassifier

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


def plot_correlation(
        array: np.ndarray | list[np.ndarray],
        title: str | list[str],
        n_cols: int = 1,
        fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the correlation matrix of the given arrays.

    Parameters
    ----------
    array: np.ndarray | list[np.ndarray]
        Input array (or list of arrays) with shape (n_channels, n_samples).
    title: str | list[str]
        Title (or list of titles) for the plot (or subplots).
    n_cols: int, default=1
        Number of columns in the plot.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    if isinstance(array, list):  # list of arrays
        # Compute n. of rows
        n_plots = len(array)
        mod = n_plots % n_cols
        n_rows = n_plots // n_cols if mod == 0 else n_plots // n_cols + mod

        for i, (a, t) in enumerate(zip(array, title)):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(np.corrcoef(a))
            plt.title(t)
            plt.grid(None)

        plt.tight_layout()

    else:  # single array
        plt.imshow(np.corrcoef(array))
        plt.title(title)
        plt.grid(None)

    plt.show()


def raster_plot(
        firings: pd.DataFrame | list[pd.DataFrame],
        title: str | list[str],
        sig_len: float,
        n_cols: int = 1,
        fig_size: tuple[int, int] | None = None
) -> None:
    """Plot a raster plot of the firing activity of a group of neurons.

    Parameters
    ----------
    firings: pd.DataFrame | list[pd.DataFrame]
        A DataFrame (or a list of DataFrames) with columns "MU index", "Firing time" and "Firing rate"
        describing the firing activity of neurons.
    title: str | list[str]
        Title (or list of titles) for the plot (or subplots).
    sig_len: float
        Length of the signal (in seconds).
    n_cols: int, default=1
        Number of columns in the plot.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    if isinstance(firings, list):  # list of DataFrames
        # Compute n. of rows
        n_plots = len(firings)
        mod = n_plots % n_cols
        n_rows = n_plots // n_cols if mod == 0 else n_plots // n_cols + mod

        for i, (f, t) in enumerate(zip(firings, title)):
            plt.subplot(n_rows, n_cols, i + 1)
            g = sns.scatterplot(
                data=f,
                x="Firing time",
                y="MU index",
                hue="Firing rate",
                palette="flare"
            )
            g.set(title=t)
            g.set(xlim=(0, sig_len))
            g.set(ylim=(-1, f["MU index"].max() + 1))

            # Color bar
            norm = plt.Normalize(f["Firing rate"].min(), f["Firing rate"].max())
            sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
            sm.set_array([])
            g.get_legend().remove()
            g.figure.colorbar(sm)

        plt.tight_layout()

    else:  # single DataFrame
        g = sns.scatterplot(
            data=firings,
            x="Firing time",
            y="MU index",
            hue="Firing rate",
            palette="flare"
        )
        g.set(title=title)
        g.set(xlim=(0, sig_len))
        g.set(ylim=(-1, firings["MU index"].max() + 1))

        # Color bar
        norm = plt.Normalize(firings["Firing rate"].min(), firings["Firing rate"].max())
        sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
        sm.set_array([])
        g.get_legend().remove()
        g.figure.colorbar(sm)

    plt.show()


def plot_connectivity(
        muapt_classifier: MUAPTClassifier,
        fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the neural connectivity of a given synapse.

    Parameters
    ----------
    muapt_classifier: MUAPTClassifier
        Instance of MUAPTClassifier.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    n_s = muapt_classifier.n_in
    n_t = muapt_classifier.n_out
    conn = list(product(range(n_s), range(n_t)))
    conn_i, conn_j = zip(*conn)

    if fig_size is not None:
        plt.figure(figsize=fig_size)
    plt.subplot(121)
    plt.plot(np.zeros(n_s), np.arange(n_s), "ok", ms=10)
    plt.plot(np.ones(n_t), np.arange(n_t), "ok", ms=10)
    for i, j in zip(conn_i, conn_j):
        plt.plot([0, 1], [i, j], "-k")
    plt.xticks([0, 1], ["Source", "Target"])
    plt.ylabel("Neuron index")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(n_s, n_t))
    plt.subplot(122)
    plt.plot(conn_i, conn_j, "ok")
    plt.xlim(-1, n_s)
    plt.ylim(-1, n_t)
    plt.xlabel("Source neuron index")
    plt.ylabel("Target neuron index")

    plt.show()


def plot_snn_hist(
        hist: dict[str, dict[Any, Any]],
        fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the training results.

    Parameters
    ----------
    hist: dict[str, dict[Any, Any]]
        Dictionary containing the SNN training/inference history.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    plt.subplot(221)
    plt.title("Input spikes")
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    for k in hist["in_spikes"].keys():
        plt.plot(hist["in_spikes"][k]["t"], hist["in_spikes"][k]["i"], ".", label=f"Gesture {k}")
    plt.legend(loc="best")

    plt.subplot(222)
    plt.title("Output spikes")
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    for k in hist["out_spikes"].keys():
        plt.plot(hist["out_spikes"][k]["t"], hist["out_spikes"][k]["i"], ".", label=f"Gesture {k}")
    plt.legend(loc="best")

    plt.subplot(223)
    plt.title("Synapses weights")
    plt.xlabel("Time (s)")
    plt.ylabel("w / g_max")
    plt.plot(hist["syn_w"]["t"], hist["syn_w"]["w"])

    plt.subplot(224)
    plt.title("Membrane potential")
    plt.xlabel("Time (s)")
    plt.ylabel("V")
    plt.plot(hist["out_v"]["t"], hist["out_v"]["V"][:, 0], label="Neuron 0")
    plt.plot(hist["out_v"]["t"], hist["out_v"]["V"][:, 1], label="Neuron 1")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
