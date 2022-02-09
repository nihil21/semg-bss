from .dataset import load_1dof
from .decomposition import fast_ica
from .fft import signal_fft
from .metrics import silhouette
from .online import garbage_detection
from .plotting import plot_signal, plot_signals
from .postprocessing import spike_detection, replicas_removal
from .preprocessing import extend_signal, whiten_signal

__all__ = [
    "extend_signal",
    "fast_ica",
    "garbage_detection",
    "load_1dof",
    "plot_signal",
    "plot_signals",
    "replicas_removal",
    "signal_fft",
    "silhouette",
    "spike_detection",
    "whiten_signal",
]
