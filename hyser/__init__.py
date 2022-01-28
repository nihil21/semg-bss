from .dataset import load_1dof
from .fast_ica import fast_ica
from .plotting import plot_signals
from .postprocessing import spike_detection, replicas_removal
from .preprocessing import extend_signal, whiten_signal

__all__ = [
    "extend_signal",
    "fast_ica",
    "load_1dof",
    "plot_signals",
    "replicas_removal",
    "spike_detection",
    "whiten_signal",
]
