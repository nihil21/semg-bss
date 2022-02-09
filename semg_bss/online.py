import numpy as np


def garbage_detection(x: np.ndarray, k: float) -> bool:
    """Determine whether the current signal is garbage or not, based on its standard deviation.

    Parameters
    ----------
    x: np.ndarray
        Input signal with shape (n_channels, n_samples).
    k: float
        Sensitivity of garbage detection algorithm.

    Returns
    -------
    garbage: bool
        Whether the signal is garbage or not.
    """

    # 1. Center signal
    x_mean = np.mean(x, axis=1, keepdims=True)
    x_center = x - x_mean
    garbage = not(all([
        x_center[i].max() < k * x_center[i].std()
        for i in range(x_center.shape[0])
    ]))
    return garbage
