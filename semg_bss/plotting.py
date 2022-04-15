from __future__ import annotations
from audioop import avg

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

from .snn import MUAPTClassifier

sns.set_theme()


def plot_signal(
    s: np.ndarray,
    fs: float = 1,
    title: str | None = None,
    labels: list[tuple[int, int, int]] | None = None,
    fig_size: tuple[int, int] | None = None
) -> None:
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s: np.ndarray
        Signal with shape (n_channels, n_samples).
    fs: float, default=1
        Sampling frequency of the signal.
    title: str | None, default=None
        Title of the whole plot.
    labels: list[tuple[int, int, int]] | None, default=None
        List containing, for each action block, the label of the action together with the first and the last samples.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    n_channels, n_samples = s.shape
    x = np.arange(n_samples) / fs

    if labels is not None:
        # Get set of unique labels
        label_set = set(map(lambda t: t[0], labels))
        # Create dictionary label -> color
        cmap = plt.cm.get_cmap("tab10", len(label_set))
        color_dict = {}
        for i, label in enumerate(label_set):
            color_dict[label] = cmap(i)

    # Create figure with subplots and shared x axis
    n_cols = 1
    n_rows = n_channels
    _, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True)

    for i in range(n_channels):
        if labels is not None:
            for label, idx_from, idx_to in labels:
                ax[i].plot(x[idx_from:idx_to], s[i, idx_from:idx_to], color=color_dict[label])
        else:
            ax[i].plot(x, s[i])
        ax[i].set_ylabel("Voltage [mV]")
    plt.xlabel("Time [s]")

    # Create legend
    if labels is not None:
        plt.legend(
            handles=[mpatches.Patch(color=c, label=l) for l, c in color_dict.items()],
            loc="upper right"
        )

    if title is not None:
        plt.suptitle(title, fontsize="xx-large")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_fft_spectrum(
    s: np.ndarray,
    fs: float = 1,
    title: str | None = None,
    fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the FFT spectrum of the input signal.

    Parameters
    ----------
    s: np.ndarray
        Input signal with shape (n_channels, n_samples).
    fs: float, default=1
        Sampling frequency of the signal.
    title: str | None, default=None
        Title of the whole plot.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    # Compute FFT spectrum
    n_channels, n_samples = s.shape
    spectrum_len = n_samples // 2
    spectrum = np.zeros(shape=(n_channels, spectrum_len), dtype=float)
    for i in range(n_channels):
        cur_spectrum = np.abs(np.fft.fft(s[i])) / n_samples
        cur_spectrum = cur_spectrum[range(spectrum_len)]
        spectrum[i] = cur_spectrum
    x = np.arange(spectrum_len) / n_samples * fs

    # Create figure with subplots and shared x axis
    n_cols = 1
    n_rows = n_channels
    _, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True)

    for i in range(n_channels):
        ax[i].plot(x, spectrum[i])
        ax[i].set_ylabel("Voltage [mV]")
    plt.xlabel("Frequency (Hz)")

    if title is not None:
        plt.suptitle(title, fontsize="xx-large")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_correlation(
    array: np.ndarray | list[np.ndarray],
    title: str | list[str],
    n_cols: int = 1,
    fig_size: tuple[int, int] | None = None,
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


def _single_raster_plot(
        firings: pd.DataFrame,
        title: str,
        sig_span: tuple[float, float],
        sort_by_negentropy: bool
) -> None:
    if sort_by_negentropy:
        g = sns.scatterplot(
            data=firings,
            x="Firing time",
            y="MU index",
            hue="Neg-entropy",
            palette="flare",
        )
        g.set(title=title)
        g.set(xlim=sig_span)
        g.set(ylim=(-1, firings["MU index"].max() + 1))
        g.set(yticks=np.arange(firings["MU index"].max()))

        # Color bar
        norm = plt.Normalize(firings["Neg-entropy"].min(), firings["Neg-entropy"].max())
        sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
        sm.set_array([])
        g.get_legend().remove()
        g.figure.colorbar(sm)
    else:
        g = sns.scatterplot(
            data=firings,
            x="Firing time",
            y="MU index",
            hue="Firing rate",
            palette="flare",
        )
        g.set(title=title)
        g.set(xlim=sig_span)
        g.set(ylim=(-1, firings["MU index"].max() + 1))
        g.set(yticks=np.arange(firings["MU index"].max()))

        # Color bar
        norm = plt.Normalize(firings["Firing rate"].min(), firings["Firing rate"].max())
        sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
        sm.set_array([])
        g.get_legend().remove()
        g.figure.colorbar(sm)


def raster_plot(
    firings: pd.DataFrame | list[pd.DataFrame],
    title: str | list[str],
    sig_span: tuple[float, float],
    sort_by_negentropy: bool = False,
    n_cols: int = 1,
    fig_size: tuple[int, int] | None = None,
) -> None:
    """Plot a raster plot of the firing activity of a group of neurons.

    Parameters
    ----------
    firings: pd.DataFrame | list[pd.DataFrame]
        A DataFrame (or a list of DataFrames) with columns "MU index", "Firing time" and "Firing rate"
        describing the firing activity of neurons.
    title: str | list[str]
        Title (or list of titles) for the plot (or subplots).
    sig_span: tuple[float, float]
        Start and end of the signal (in seconds).
    sort_by_negentropy: bool, default=False
        Whether to sort MUs by neg-entropy or firing rate (default).
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

        ax = None
        for i, (f, t, sp) in enumerate(zip(firings, title, sig_span)):
            if i % n_cols == 0:  # new row
                ax = plt.subplot(n_rows, n_cols, i + 1)
            else:  # same row as before
                ax = plt.subplot(n_rows, n_cols, i + 1, sharey=ax)

            # Draw plot
            _single_raster_plot(f, t, sp, sort_by_negentropy)
        plt.tight_layout()

    else:  # single DataFrame
        # Draw plot
        _single_raster_plot(firings, title, sig_span, sort_by_negentropy)

    plt.show()


def plot_classifier_hist(
    history: dict[int, tf.keras.callbacks.History],
    validation: bool = False,
    title: str | None = None,
    fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the training and validation history of a classifier.
    
    Parameters
    ----------
    history: dict[int, tf.keras.callbacks.History]
        Dictionary containing, for each run, a Keras History object.
    validation: bool, default=False
        Whether to plot the validation history.
    title: str | None, default=None
        Title of the whole plot.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    if fig_size:
        plt.figure(figsize=fig_size)
    
    n_rows = 2 if validation else 1

    # Get maximum number of epochs across runs
    ep_runs = [len(h.history["loss"]) for _, h in history.items()]
    min_ep = min(ep_runs)

    # Training loss
    ax_loss = plt.subplot(n_rows, 2, 1)
    plt.title("Training loss")
    plt.xlabel("Epochs")
    
    avg_train_loss = np.zeros(shape=(0, min_ep))
    for i, h in history.items():
        arr = np.array(h.history["loss"])
        plt.plot(arr, alpha=0.3, label=f"Run {i + 1}")
        avg_train_loss = np.concatenate([avg_train_loss, arr.reshape(1, -1)[:, :min_ep]])
    
    avg_train_loss = avg_train_loss.mean(axis=0)
    plt.plot(avg_train_loss, label=f"Average across runs")
    plt.legend()

    # Training accuracy
    ax_acc = plt.subplot(n_rows, 2, 2)
    plt.title("Training accuracy")
    plt.xlabel("Epochs")

    avg_train_acc = np.zeros(shape=(0, min_ep))
    for i, h in history.items():
        arr = np.array(h.history["accuracy"])
        plt.plot(arr, alpha=0.3, label=f"Run {i + 1}")
        avg_train_acc = np.concatenate([avg_train_acc, arr.reshape(1, -1)[:, :min_ep]])
    
    avg_train_acc = avg_train_acc.mean(axis=0)
    plt.plot(avg_train_acc, label=f"Average across runs")
    plt.legend()

    if validation:
        # Validation loss
        plt.subplot(n_rows, 2, 3, sharex=ax_loss)
        plt.title("Validation loss")
        plt.xlabel("Epochs")

        avg_val_loss = np.zeros(shape=(0, min_ep))
        for i, h in history.items():
            arr = np.array(h.history["val_loss"])
            plt.plot(arr, alpha=0.3, label=f"Run {i + 1}")
            avg_val_loss = np.concatenate([avg_val_loss, arr.reshape(1, -1)[:, :min_ep]])
        
        avg_val_loss = avg_val_loss.mean(axis=0)
        plt.plot(avg_val_loss, label=f"Average across runs")
        plt.legend()

        # Validation accuracy
        plt.subplot(n_rows, 2, 4, sharex=ax_acc)
        plt.title("Validation accuracy")
        plt.xlabel("Epochs")

        avg_val_acc = np.zeros(shape=(0, min_ep))
        for i, h in history.items():
            arr = np.array(h.history["val_accuracy"])
            plt.plot(arr, alpha=0.3, label=f"Run {i + 1}")
            avg_val_acc = np.concatenate([avg_val_acc, arr.reshape(1, -1)[:, :min_ep]])
        
        avg_val_acc = avg_val_acc.mean(axis=0)
        plt.plot(avg_val_acc, label=f"Average across runs")
        plt.legend()
    
    if title is not None:
        plt.suptitle(title, fontsize="xx-large")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_connectivity(
    muapt_classifier: MUAPTClassifier, fig_size: tuple[int, int] | None = None
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
    hist: dict[str, dict[Any, Any]], fig_size: tuple[int, int] | None = None
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
        plt.plot(
            hist["in_spikes"][k]["t"],
            hist["in_spikes"][k]["i"],
            ".",
            label=f"Gesture {k}",
        )
    plt.legend(loc="best")

    plt.subplot(222)
    plt.title("Output spikes")
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    for k in hist["out_spikes"].keys():
        plt.plot(
            hist["out_spikes"][k]["t"],
            hist["out_spikes"][k]["i"],
            ".",
            label=f"Gesture {k}",
        )
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
    plt.plot(hist["out_v"]["t"], hist["out_v"]["v"][:, 0], label="Neuron 0")
    plt.plot(hist["out_v"]["t"], hist["out_v"]["v"][:, 1], label="Neuron 1")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
