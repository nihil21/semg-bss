from typing import Optional

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


def pearson_corr(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Given two matrices with shape (n_features, n_samples), compute the Pearson correlation matrix.

    Parameters
    ----------
    x1: np.ndarray
        First matrix with shape (n_features, n_samples).
    x2: np.ndarray
        First matrix with shape (n_features, n_samples).

    Returns
    -------
    corr_mtx: np.ndarray
        Correlation matrix with shape (n_features, n_features).
    """
    corr_mtx = np.zeros(shape=(x1.shape[0], x2.shape[0]), dtype=float)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            corr_mtx[i, j] = np.corrcoef(x1[i], x2[j])[0, 1]
    return corr_mtx


def permutation(x_cur: np.ndarray, x_prev: np.ndarray) -> Optional[np.ndarray]:
    """Compute the permutation matrix by comparing data in the current and previous time windows.

    Parameters
    ----------
    x_cur: np.ndarray
        Data in the current time window.
    x_prev: np.ndarray
        Data in the previous time window.

    Returns
    -------
    perm_mtx: Optional[np.ndarray]
        Permutation matrix or None, if the one-to-one mapping constraint fails.
    """

    # Compute correlation matrix
    corr_mtx = pearson_corr(x_cur, x_prev)
    # Compute row-wise max of absolute correlation matrix
    row_max = np.amax(np.abs(corr_mtx), axis=1)

    # Build permutation matrix
    perm_mtx = np.zeros(shape=corr_mtx.shape, dtype=int)
    for i in range(corr_mtx.shape[0]):
        for j in range(corr_mtx.shape[1]):
            if np.abs(corr_mtx[i, j]) == row_max[i] and corr_mtx[i, j] >= 0:
                perm_mtx[i, j] = 1
            elif np.abs(corr_mtx[i, j]) == row_max[i] and corr_mtx[i, j] < 0:
                perm_mtx[i, j] = -1

    # Check one-to-one mapping
    one_to_one = np.all(np.abs(perm_mtx).sum(axis=0) == 1)

    return perm_mtx if one_to_one else None
