from .dataset import load_1dof
from .decomposition import fast_ica
from .metrics import silhouette
from .plotting import plot_signals
from .postprocessing import spike_detection, replicas_removal
from .preprocessing import extend_signal, whiten_signal

__all__ = [
    "extend_signal",
    "fast_ica",
    "load_1dof",
    "plot_signals",
    "replicas_removal",
    "silhouette",
    "spike_detection",
    "whiten_signal",
]
