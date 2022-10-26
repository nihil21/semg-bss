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

from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches as m_patches
from matplotlib import pyplot as plt

sns.set_theme()
# sns.set(font_scale=1.5)


def plot_signal(
    s: np.ndarray,
    fs: float = 1,
    title: str | None = None,
    labels: list[tuple[int, int, int]] | None = None,
    resolution: int | None = None,
    fig_size: tuple[int, int] | None = None
) -> None:
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s : ndarray
        Signal with shape (n_channels, n_samples).
    fs : float, default=1
        Sampling frequency of the signal.
    title : str | None, default=None
        Title of the whole plot.
    labels : list of tuple of (int, int, int) | None, default=None
        List containing, for each action block, the label of the action together with the first and the last samples.
    resolution : int | None, default=None
        Resolution for the x-axis.
    fig_size : tuple of (int, int) | None, default=None
        Height and width of the plot.
    """
    n_channels, n_samples = s.shape
    x = np.arange(n_samples) / fs

    color_dict = {}
    if labels is not None:
        # Get set of unique labels
        label_set = set(map(lambda t: t[0], labels))
        # Create dictionary label -> color
        cmap = plt.cm.get_cmap("tab10", len(label_set))
        for i, label in enumerate(label_set):
            color_dict[label] = cmap(i)

    # Create figure with subplots and shared x-axis
    n_cols = 1
    n_rows = n_channels
    _, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex="all")

    if labels is not None:
        if n_channels == 1:
            for label, idx_from, idx_to in labels:
                ax.plot(x[idx_from:idx_to], s[0, idx_from:idx_to], color=color_dict[label])
            ax.set_ylabel("Voltage [mV]")
        else:
            for i in range(n_channels):
                for label, idx_from, idx_to in labels:
                    ax[i].plot(x[idx_from:idx_to], s[i, idx_from:idx_to], color=color_dict[label])
                ax[i].set_ylabel("Voltage [mV]")
    else:
        if n_channels == 1:
            ax.plot(x, s[0])
            ax.set_ylabel("Voltage [mV]")
        else:
            for i in range(n_channels):
                ax[i].plot(x, s[i])
                ax[i].set_ylabel("Voltage [mV]")
    plt.xlabel("Time [s]")

    if resolution is not None:
        plt.xticks(range(0, n_samples // fs, resolution))

    # Create legend
    if labels is not None:
        plt.legend(
            handles=[m_patches.Patch(color=c, label=lab) for lab, c in color_dict.items()],
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
    s : ndarray
        Input signal with shape (n_channels, n_samples).
    fs : float, default=1
        Sampling frequency of the signal.
    title : str | None, default=None
        Title of the whole plot.
    fig_size : tuple of (int, int) | None, default=None
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

    # Create figure with subplots and shared x-axis
    n_cols = 1
    n_rows = n_channels
    _, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex="all")

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
    title: str | list[str] | None = None,
    n_cols: int = 1,
    fig_size: tuple[int, int] | None = None,
) -> None:
    """Plot the correlation matrix of the given arrays.

    Parameters
    ----------
    array : ndarray | list of ndarray
        Input array (or list of arrays) with shape (n_channels, n_samples).
    title : str | list of str | None
        Title (or list of titles) for the plot (or subplots).
    n_cols : int, default=1
        Number of columns in the plot.
    fig_size : tuple of (int, int) | None, default=None
        Height and width of the plot.
    """
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    if isinstance(array, list):  # list of arrays
        # Compute n. of rows
        n_plots = len(array)
        mod = n_plots % n_cols
        n_rows = n_plots // n_cols if mod == 0 else n_plots // n_cols + mod

        if title is None:
            title = [None] * len(array)
        for i, (a, t) in enumerate(zip(array, title)):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(np.corrcoef(a))
            if t is not None:
                plt.title(t)
            plt.grid(None)

        plt.tight_layout()

    else:  # single array
        plt.imshow(np.corrcoef(array))
        if title is not None:
            plt.title(title)
        plt.grid(None)

    plt.show()


def _single_raster_plot(
        firings: pd.DataFrame,
        title: str | None,
        sig_span: tuple[float, float],
        negentropy_hue: bool
) -> None:
    if negentropy_hue:
        g = sns.scatterplot(
            data=firings,
            x="Firing time",
            y="MU index",
            hue="Neg-entropy",
            palette="flare"
        )
        if title is not None:
            g.set(title=title)
        g.set(xlim=sig_span)
        g.set(xlabel="Time [s]")
        g.set_yticks = range(firings["MU index"].max())

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
            palette="flare"
        )
        if title is not None:
            g.set(title=title)
        g.set(xlim=sig_span)
        g.set(xlabel="Time [s]")
        g.set_yticks = range(firings["MU index"].max())

        # Color bar
        norm = plt.Normalize(firings["Firing rate"].min(), firings["Firing rate"].max())
        sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
        sm.set_array([])
        g.get_legend().remove()
        g.figure.colorbar(sm)


def raster_plot(
    firings: pd.DataFrame | list[pd.DataFrame],
    sig_span: tuple[float, float] | list[tuple[float, float]],
    title: str | list[str] | None = None,
    negentropy_hue: bool = False,
    n_cols: int = 1,
    fig_size: tuple[int, int] | None = None,
) -> None:
    """Plot a raster plot of the firing activity of a group of neurons.

    Parameters
    ----------
    firings : DataFrame | list of DataFrame
        A DataFrame (or a list of DataFrames) with columns "MU index", "Firing time" and "Firing rate"
        describing the firing activity of neurons.
    sig_span : tuple of (float, float) | list of tuple of (float, float)
        Start and end of the signal (in seconds).
    title : str | list of str | None, default=None
        Title (or list of titles) for the plot (or subplots).
    negentropy_hue : bool, default=False
        Whether to use the neg-entropy or firing rate (default) for coloring the scatter plot.
    n_cols : int, default=1
        Number of columns in the plot.
    fig_size : tuple of (int, int) | None, default=None
        Height and width of the plot.
    """
    if fig_size is not None:
        plt.figure(figsize=fig_size)

    if isinstance(firings, list) and isinstance(sig_span, list) \
            and (isinstance(title, list) or title is None):  # list of DataFrames
        # Compute n. of rows
        n_plots = len(firings)
        mod = n_plots % n_cols
        n_rows = n_plots // n_cols if mod == 0 else n_plots // n_cols + mod

        ax = None

        if title is None:
            title = [None] * len(firings)
        for i, (f, t, sp) in enumerate(zip(firings, title, sig_span)):
            if i % n_cols == 0:  # new row
                ax = plt.subplot(n_rows, n_cols, i + 1)
            else:  # same row as before
                ax = plt.subplot(n_rows, n_cols, i + 1, sharey=ax)

            # Draw plot
            _single_raster_plot(f, t, sp, negentropy_hue)
        plt.tight_layout()

    else:  # single DataFrame
        # Draw plot
        _single_raster_plot(firings, title, sig_span, negentropy_hue)
    
    plt.show()


def plot_firings_comparison(
    firings1: pd.DataFrame,
    firings2: pd.DataFrame,
    sig_span: tuple[float, float],
    fig_size: tuple[int, int] | None = None,
) -> None:
    """Plot two raster plots of the firing activity of a group of neurons one above the other, for comparison.

    Parameters
    ----------
    firings1 : DataFrame
        A DataFrame with columns "MU index" and "Firing time" describing the firing activity of neurons.
    firings2 : DataFrame
        A DataFrame with columns "MU index" and "Firing time" describing the firing activity of neurons.
    sig_span : tuple of (float, float) | list of tuple of (float, float)
        Start and end of the signal (in seconds).
    fig_size : tuple of (int, int) | None, default=None
        Height and width of the plot.
    """
    min_n_mu = min(
        (firings1["MU index"].max(), firings2["MU index"].max())
    )

    if fig_size:
        plt.figure(figsize=fig_size)
    plt.xlabel("Time [s]")
    plt.xlim(sig_span)
    plt.ylabel("MU index")
    plt.yticks(range(0, min_n_mu + 1, 1))
    plt.hlines(np.arange(-0.5, min_n_mu + 1, 1), xmin=sig_span[0], xmax=sig_span[1], colors="k", linestyles="dashed")

    for cur_f_s1 in firings1[firings1["MU index"] <= min_n_mu].groupby("MU index")["Firing time"]:
        plt.scatter(
            x=cur_f_s1[1].values,
            y=[cur_f_s1[0] + 0.18] * len(cur_f_s1[1].values),
            marker="|",
            color="b",
            s=80
        )
    
    for cur_f_s2 in firings2[firings2["MU index"] <= min_n_mu].groupby("MU index")["Firing time"]:
        plt.scatter(
            x=cur_f_s2[1].values,
            y=[cur_f_s2[0] - 0.18] * len(cur_f_s2[1].values),
            marker="|",
            color="r",
            s=80
        )

    plt.legend(
        handles=[m_patches.Patch(color="b", label="Session 1"), m_patches.Patch(color="r", label="Session 2")]
    )
    
    plt.show()


def plot_classifier_hist(
    history: dict[int, dict[str, list[float]]],
    validation: bool = False,
    title: str | None = None,
    fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the training and validation history of a classifier.
    
    Parameters
    ----------
    history : dict of {int, dict of {str, list of float}}
        Dictionary containing, for each run, another dictionary with the training history.
    validation : bool, default=False
        Whether to plot the validation history.
    title : str | None, default=None
        Title of the whole plot.
    fig_size : tuple of (int, int) | None, default=None
        Height and width of the plot.
    """
    if fig_size:
        plt.figure(figsize=fig_size)
    
    n_rows = 2 if validation else 1

    # Get maximum number of epochs across runs
    ep_runs = [len(h["loss"]) for _, h in history.items()]
    min_ep = min(ep_runs)

    # Training loss
    ax_loss = plt.subplot(n_rows, 2, 1)
    plt.title("Training loss")
    plt.xlabel("Epochs")
    
    avg_train_loss = np.zeros(shape=(0, min_ep))
    for i, h in history.items():
        arr = np.array(h["loss"])
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
        arr = np.array(h["accuracy"])
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
            arr = np.array(h["val_loss"])
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
            arr = np.array(h["val_accuracy"])
            plt.plot(arr, alpha=0.3, label=f"Run {i + 1}")
            avg_val_acc = np.concatenate([avg_val_acc, arr.reshape(1, -1)[:, :min_ep]])
        
        avg_val_acc = avg_val_acc.mean(axis=0)
        plt.plot(avg_val_acc, label=f"Average across runs")
        plt.legend()
    
    if title is not None:
        plt.suptitle(title, fontsize="xx-large")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_snn_hist(
    hist: dict[str, dict[Any, Any]], fig_size: tuple[int, int] | None = None
) -> None:
    """Plot the training results.

    Parameters
    ----------
    hist : dict of {str, dict}
        Dictionary containing the SNN training/inference history.
    fig_size : tuple of (int, int) | None, default=None
        Height and width of the plot.
    """
    _, ax = plt.subplots(5, 1, figsize=fig_size, sharex="all")

    ax[0].set_title("Input spikes")
    ax[0].set_ylabel("Neuron index")
    for k in hist["spikes_inp"].keys():
        ax[0].plot(
            hist["spikes_inp"][k]["t"],
            hist["spikes_inp"][k]["i"],
            ".",
            label=f"Gesture {k}",
        )
    ax[0].legend(loc="best")

    ax[1].set_title("Spikes of excitatory neurons")
    ax[1].set_ylabel("Neuron index")
    for k in hist["spikes_exc"].keys():
        ax[1].plot(
            hist["spikes_exc"][k]["t"],
            hist["spikes_exc"][k]["i"],
            ".",
            label=f"Gesture {k}",
        )
    ax[1].legend(loc="best")

    ax[2].set_title("Spikes of inhibitory neurons")
    ax[2].set_ylabel("Neuron index")
    for k in hist["spikes_inh"].keys():
        ax[2].plot(
            hist["spikes_inh"][k]["t"],
            hist["spikes_inh"][k]["i"],
            ".",
            label=f"Gesture {k}",
        )
    ax[2].legend(loc="best")

    ax[3].set_title("Membrane potential of excitatory neurons")
    ax[3].set_ylabel("V")
    ax[3].plot(hist["V_exc"]["t"], hist["V_exc"]["V"])
    ax[3].plot(hist["V_exc"]["t"], hist["V_exc"]["thres"], linestyle="dashed")

    ax[4].set_title("Membrane potential of inhibitory neurons")
    ax[4].set_ylabel("V")
    ax[4].plot(hist["V_inh"]["t"], hist["V_inh"]["V"])
    ax[4].plot(hist["V_inh"]["t"], hist["V_inh"]["thres"], linestyle="dashed")

    # ax[5].set_title("Synaptic weights (from input to excitatory)")
    # ax[5].set_ylabel("w")
    # ax[5].plot(hist["w_inp2exc"]["t"], hist["w_inp2exc"]["w"])

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
