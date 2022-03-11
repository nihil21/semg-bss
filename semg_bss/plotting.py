import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from matplotlib import pyplot as plt
#
#
# def plot_signal(
#         s: np.ndarray,
#         fs: float = 1,
#         n_cols: int = 1,
#         fig_size: Optional[Tuple[int, int]] = None
# ):
#     """Plot a signal with multiple channels.
#
#     Parameters
#     ----------
#     s: np.ndarray
#         Signal with shape (n_channels, n_samples).
#     fs: float, default=1
#         Sampling frequency of the signal.
#     n_cols: int, default=1
#         Number of columns in the plot.
#     fig_size: Optional[Tuple[int, int]], default=None
#         Size of Matplotlib figure.
#     """
#
#     if fig_size:
#         plt.figure(figsize=fig_size)
#     n_channels, n_samples = s.shape
#     x = np.arange(n_samples) / fs
#
#     # Compute n. of rows
#     mod = n_channels % n_cols
#     n_rows = n_channels // n_cols if mod == 0 else n_channels // n_cols + mod
#
#     for i in range(n_channels):
#         plt.subplot(n_rows, n_cols, i + 1)
#         plt.title(f"Channel {i + 1}")
#         plt.plot(x, s[i], "tab:blue")
#         plt.grid()
#
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_signals(
#         s1: np.ndarray,
#         s2: np.ndarray,
#         fs: float = 1,
#         n_cols: int = 1,
#         fig_size: Optional[Tuple[int, int]] = None
# ):
#     """Plot side by side multiple signals with multiple channels.
#
#     Parameters
#     ----------
#     s1: np.ndarray
#         First signals with shape (n_channels, n_samples).
#     s2: np.ndarray
#         Second signals with shape (n_channels, n_samples).
#     fs: float, default=1
#         Sampling frequency of the two signals.
#     n_cols: int, default=1
#         Number of columns in the plot.
#     fig_size: Optional[Tuple[int, int]], default=None
#         Size of Matplotlib figure.
#     """
#     assert s1.shape[0] == s2.shape[0], "The two signals must have the same number of channels."
#
#     if fig_size:
#         plt.figure(figsize=fig_size)
#     n_channels, n_samples1 = s1.shape
#     _, n_samples2 = s2.shape
#     x1 = np.arange(n_samples1) / fs
#     x2 = np.arange(n_samples2) / fs
#
#     # Compute n. of rows for one signal
#     mod = n_channels % n_cols
#     n_rows = n_channels // n_cols if mod == 0 else n_channels // n_cols + 1
#     offset = 0
#     for i in range(n_rows):
#         term = n_cols
#         # Adjust for last row if mod != 0
#         if i == n_rows - 1 and mod != 0:
#             term = mod
#
#         for j in range(term):
#             idx1 = 2 * offset + j + 1
#             plt.subplot(2 * n_rows, n_cols, idx1)
#             plt.title(f"Signal 1, channel {offset + j + 1}")
#             plt.plot(x1, s1[offset + j], "tab:blue")
#             plt.grid()
#             idx2 = 2 * offset + n_cols + j + 1
#             plt.subplot(2 * n_rows, n_cols, idx2)
#             plt.title(f"Signal 2, channel {offset + j + 1}")
#             plt.plot(x2, s2[offset + j], "tab:orange")
#             plt.grid()
#         offset += n_cols
#
#     plt.tight_layout()
#     plt.show()


def plot_signal(
        s: np.ndarray,
        fig_size: tuple[int, int],
        fs: float = 1,
        n_cols: int = 1
):
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s: np.ndarray
        Signal with shape (n_channels, n_samples).
    fig_size: tuple[int, int]
        Size of figure.
    fs: float, default=1
        Sampling frequency of the signal.
    n_cols: int, default=1
        Number of columns in the plot.
    """
    n_channels, n_samples = s.shape
    x = np.arange(n_samples) / fs

    # Compute n. of rows
    mod = n_channels % n_cols
    n_rows = n_channels // n_cols if mod == 0 else n_channels // n_cols + mod

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"Channel {i}" for i in range(n_channels)])

    for i in range(n_channels):
        fig.add_trace(
            go.Scatter(x=x, y=s[i]),
            row=i // n_cols + 1,
            col=i % n_cols + 1
        )

    fig.update_layout(height=fig_size[0], width=fig_size[1])
    fig.show()


def raster_plot(f: pd.DataFrame, fig_size: tuple[int, int]):
    """Plot a raster plot of the firing activity of a group of neurons.

    Parameters
    ----------
    f: pd.DataFrame
        A DataFrame with columns "MU index", "Firing time" and "Firing rate" describing the firing activity of neurons.
    fig_size: tuple[int, int]
        Size of figure.
    """
    # mu_idx = f["MU index"].unique()
    # mapping = {idx: i for i, idx in enumerate(mu_idx)}
    # f["Firing rate idx"] = f["MU index"].apply(lambda x: mapping[x])
    fig = px.scatter(
        f,
        x="Firing time",
        y="MU index",
        color="Firing rate",
        range_x=[0, 25],
    )

    fig.update_layout(height=fig_size[0], width=fig_size[1], title_text="Neuronal activity")
    fig.show()
