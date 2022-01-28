import time
from typing import Callable, Optional, Tuple

import numpy as np


def _logcosh(x: np.ndarray):
    """LogCosh function.

    Parameters
    ----------
    x: np.ndarray
        Input data.

    Returns
    -------
    gx: np.ndarray
        LogCosh output.
    gx_prime: np.ndarray
        LogCosh derivative.
    """

    alpha = 1.0
    # Compute G
    gx = np.tanh(alpha * x)
    # Compute G'
    gx_prime = alpha * (1 - gx ** 2)

    return gx, gx_prime


def _exp(x: np.ndarray):
    """Exp function.

    Parameters
    ----------
    x: np.ndarray
        Input data.

    Returns
    -------
    gx: np.ndarray
        Exp output.
    gx_prime: np.ndarray
        Exp derivative.
    """

    exp = np.exp(-x ** 2 / 2)
    # Compute G
    gx = x * exp
    # Compute G'
    gx_prime = (1 - x ** 2) * exp

    return gx, gx_prime


def _cube(x: np.ndarray):
    """Cubic function.

    Parameters
    ----------
    x: np.ndarray
        Input data.

    Returns
    -------
    gx: np.ndarray
        Cubic output.
    gx_prime: np.ndarray
        Cubic derivative.
    """

    # Compute G
    gx = x ** 3
    # Compute G'
    gx_prime = (3 * x ** 2)

    return gx, gx_prime


def _gram_schmidt_decorrelation(b_i_new: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gram-Schmidt decorrelation.

    Parameters
    ----------
    b_i_new: np.ndarray
        New separation vector.
    b: np.ndarray
        Old separation matrix.

    Returns
    -------
    b_i_new: np.ndarray
        New decorrelated separation vector.
    """

    b_i_new -= b @ b.T @ b_i_new
    return b_i_new / np.linalg.norm(b_i_new)


def _symmetric_decorrelation(b: np.ndarray) -> np.ndarray:
    """Symmetric decorrelation.

    Parameters
    ----------
    b: np.ndarray
        Separation matrix.

    Returns
    -------
    b: np.ndarray
        Decorrelated separation matrix.
    """

    # Compute eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eigh(np.dot(b, b.T))
    # Construct diagonal matrix of eigenvalues
    eps = 1e-10
    d = np.diag(1. / (eig_vals + eps) ** 0.5)
    # Compute new separation matrix
    return eig_vecs @ d @ eig_vecs.T @ b


def _ica_def(
        x: np.ndarray,
        n_components: int,
        g: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        max_iter: int,
        threshold: float,
        verbose: bool,
) -> np.ndarray:
    """FastICA subroutine implementing deflation strategy.

    Parameters
    ----------
    x: np.ndarray
        Input data with shape (n_channels, n_samples).
    n_components: int
        Number of components to extract.
    g: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
        Non-quadratic function G.
    max_iter: int
        Maximum number of iterations.
    threshold: float
        Threshold for FastICA convergence.
    verbose: bool
        Whether to log information or not.

    Returns
    -------
    b: np.ndarray
        Separation matrix.
    """

    n_channels, n_samples = x.shape

    # Separation matrix
    b = np.zeros(shape=(n_channels, n_components), dtype=float)

    start = time.time()
    # Iterate over channels
    for i in range(n_components):
        if verbose:
            print("\r", end="")
            print(f"Component {i + 1}/{n_components}", end="", flush=True)

        # Initialize i-th separation vector
        b_i = np.random.randn(n_channels, 1)
        b_i /= np.linalg.norm(b_i)

        # Iterate until convergence or maxIter are reached
        for _ in range(max_iter):
            # (n_channels, 1).T @ (n_channels, n_samples) -> (1, n_samples)
            g_ws, g_ws_prime = g(b_i.T @ x)
            # (n_channels, n_samples) * (1, n_samples).T -> (n_channels, 1)
            t1 = x @ g_ws.T / n_samples
            # E[(n_samples,)] * (n_channels, 1) -> (n_channels, 1)
            t2 = g_ws_prime.mean() * b_i

            # Compute new separation vector
            b_i_new = t1 - t2
            # Decorrelate
            b_i_new = _gram_schmidt_decorrelation(b_i_new, b)

            # Compute distance
            distance = np.abs(np.dot(b_i.T, b_i_new) - 1)
            # Update separation vector
            b_i = b_i_new
            # Check convergence
            if distance < threshold:
                break

        b[:, i] = b_i.squeeze()

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"FastICA done in {elapsed:.2f} s")

    return b


def _ica_par(
        x: np.ndarray,
        n_components: int,
        g: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        max_iter: int,
        threshold: float,
        verbose: bool,
) -> np.ndarray:
    """FastICA subroutine implementing parallel strategy.

    Parameters
    ----------
    x: np.ndarray
        Input data with shape (n_channels, n_samples).
    n_components: int
        Number of components to extract.
    g: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
        Non-quadratic function G.
    max_iter: int
        Maximum number of iterations.
    threshold: float
        Threshold for FastICA convergence.
    verbose: bool
        Whether to log information or not.

    Returns
    -------
    b: np.ndarray
        Separation matrix.
    """

    n_channels, n_samples = x.shape

    # Initialize separation matrix randomly and decorrelate
    b = np.random.randn(n_channels, n_components)
    b = _symmetric_decorrelation(b)

    start = time.time()
    # Iterate until convergence or maxIter are reached
    for _ in range(max_iter):
        b_new = np.zeros(shape=(n_channels, n_components), dtype=float)
        # Iterate over channels
        for i in range(n_components):
            b_i = b[:, i]
            # (n_channels, 1).T @ (n_channels, n_samples) -> (1, n_samples)
            g_ws, g_ws_prime = g(b_i.T @ x)
            # (n_channels, n_samples) * (1, n_samples).T -> (n_channels, 1)
            t1 = x @ g_ws.T / n_samples
            # E[(n_samples,)] * (n_channels, 1) -> (n_channels, 1)
            t2 = g_ws_prime.mean() * b_i
            # Compute new separation vector
            b_i_new = t1 - t2
            # Save new separation vector
            b_new[:, i] = b_i_new

        # Decorrelate
        b_new = _symmetric_decorrelation(b_new)

        # Compute distance
        distance = max(abs(abs(np.diag(np.dot(b_new, b.T))) - 1))
        # Update separation matrix
        b = b_new
        # Check convergence
        if distance < threshold:
            break

    if verbose:
        elapsed = time.time() - start
        print("\r", end="")
        print(f"FastICA done in {elapsed:.2f} s")

    return b


def fast_ica(
        x: np.ndarray,
        n_components: Optional[int] = None,
        strategy: str = "deflation",
        g_func: str = "logcosh",
        max_iter: int = 100,
        threshold: float = 1e-4,
        verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """FastICA implementation.

    Parameters
    ----------
    x: np.ndarray
        Input data with shape (n_channels, n_samples).
    n_components: Optional[int], default=None
        Number of components to extract (if None it's assumed to be equal to the number of channels).
    strategy: str, default="deflation"
        FastICA decomposition strategy ("deflation" or "parallel").
    g_func: str, default="logcosh"
        Non-quadratic function G ("logcosh", "exp" or "cube").
    max_iter: int, default=100
        Maximum number of iterations.
    threshold: float, default=1e-4
        Threshold for FastICA convergence.
    verbose: bool, default=False
        Whether to log information or not.

    Returns
    -------
    s: np.ndarray
        Recovered signal with shape (n_components, n_samples).
    b: np.ndarray
        Separation matrix.
    """

    # Strategy dictionary
    func_dict = {
        "deflation": _ica_def,
        "parallel": _ica_par
    }
    # Non-quadratic function G dictionary
    g_dict = {
        "logcosh": _logcosh,
        "exp": _exp,
        "cube": _cube
    }
    kwargs = {
        "n_components": n_components if n_components is not None else x.shape[0],
        "g": g_dict[g_func],
        "max_iter": max_iter,
        "threshold": threshold,
        "verbose": verbose,
    }

    # Compute separation matrix
    b = func_dict[strategy](x, **kwargs)

    s = b.T @ x

    return s, b
