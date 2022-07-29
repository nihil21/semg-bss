"""Copyright 2022 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks

from .preprocessing import extend_signal, center_signal, whiten_signal


def _logcosh(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        LogCosh first derivative.
    gx_sec: np.ndarray
        LogCosh second derivative.
    """

    alpha = 1.0
    # Compute G
    gx = 1 / alpha * np.log(np.cosh(alpha * x))
    # Compute G'
    gx_prime = np.tanh(alpha * x)
    # Compute G''
    gx_sec = alpha * (1 - gx_prime ** 2)

    return gx, gx_prime, gx_sec


def _exp(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    gx = -np.exp(-(x ** 2) / 2)
    # Compute G

    # Compute G'
    gx_prime = -x * gx
    # Compute G''
    gx_sec = (x ** 2 - 1) * gx

    return gx, gx_prime, gx_sec


def _skew(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skewness function.

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
    gx = x ** 3 / 3
    # Compute G'
    gx_prime = x ** 2
    # Compute G''
    gx_sec = 2 * x

    return gx, gx_prime, gx_sec


def _fast_ica_iter(
        emg_white: np.ndarray,
        wi: np.ndarray,
        g: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> np.ndarray:
    """FastICA iteration.

    Parameters
    ----------
    emg_white: np.ndarray
        Pre-whitened sEMG signal with shape (n_channels, n_samples).
    wi: np.ndarray
        Old separation vector with shape (n_channels,).
    g: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
        Contrast function.

    Returns
    -------
    wi_new: np.ndarray
        New separation vector with shape (n_channels,).
    """
    # (n_channels,) @ (n_channels, n_samples) -> (n_samples,)
    _, g_ws_prime, g_ws_sec = g(np.dot(wi, emg_white))
    # (n_channels, n_samples) * (n_samples,) -> (n_channels, 1)
    t1 = (emg_white * g_ws_prime).mean(axis=-1)
    # E[(n_samples,)] * (n_channels, 1) -> (n_channels, 1)
    t2 = g_ws_sec.mean() * wi
    # Compute new separation vector
    wi_new = t1 - t2

    return wi_new


# Dictionary for contrast functions
g_dict = {"logcosh": _logcosh, "exp": _exp, "skew": _skew}


@dataclass
class DecompositionParameters:
    """Parameters of a decomposition model."""
    mean_vec: np.ndarray | None = None
    white_mtx: np.ndarray | None = None
    sep_mtx: np.ndarray | None = None
    spike_th: np.ndarray | None = None


class EMGSeparator:
    """Perform blind source separation of sEMG signals via convolutive FastICA + source improvement.

    Parameters
    ----------
    max_sources: int
        Maximum n. of sources that can be extracted.
    samp_freq: float
        Sampling frequency of the signal.
    f_e: int, default=0
        Extension factor for the signal.
    reg_factor: float, default=0.5
        Regularization factor for whitening.
    g_func: str, default="logcosh"
        Contrast function for FastICA.
    conv_th: float, default=1e-4
        Threshold for convergence.
    sil_th: float, default=0.9
        Threshold for source acceptance based on the silhouette score.
    max_iter: int, default=100
        Maximum n. of iterations.
    min_spike_distance: float, default=10
        Minimum distance between two spikes.
    min_perc: float, default=0.5
        Minimum percentage of common firings for considering two MUs as duplicates.
    sorting_strategy: str | None, default=None
        Sorting strategy (either "firing-rate", "neg-entropy", or None).
    momentum: float, default=0.5
        Momentum update for whitening matrix and mean vector when recalibrating.
    seed: Optional[int], default=None
        Seed for the internal PRNG.

    Attributes
    ----------
    _is_calibrated: bool
        Whether the instance is calibrated or not.
    _max_sources: int
        Maximum n. of sources that can be extracted.
    _samp_freq: float
        Sampling frequency of the signal.
    _f_e: int
        Extension factor for the signal.
    _reg_factor: float
        Regularization factor for whitening.
    _g_func: str
        Name of the contrast function.
    _conv_th: float
        Threshold for convergence.
    _sil_th: float
        Threshold for source acceptance based on the silhouette score.
    _max_iter: int
        Maximum n. of iterations.
    _min_spike_distance: float
        Minimum distance between two spikes.
    _min_perc: float
        Minimum percentage of common firings for considering two MUs as duplicates.
    _sorting_strategy: str | None
        Sorting strategy (either "firing-rate", "neg-entropy", or None).
    _momentum: float
        Momentum update for whitening matrix and mean vector when recalibrating.
    _prng: np.Generator
        Actual PRNG.
    _params: DecompositionParameters
        Decomposition parameters learnt during calibration.
    _n_mu: int
        Number of extracted MUs.
    """

    def __init__(
            self,
            max_sources: int,
            samp_freq: float,
            f_e: int = 0,
            reg_factor: float = 0.5,
            g_func: str = "logcosh",
            conv_th: float = 1e-4,
            sil_th: float = 0.9,
            max_iter: int = 100,
            min_spike_distance: float = 10,
            min_perc: float = 0.5,
            sorting_strategy: str = None,
            momentum: float = 0.5,
            seed: int | None = None,
    ):
        # Parameter check
        assert max_sources > 0, "The maximum n. of sources must be positive."
        assert g_func in [
            "logcosh",
            "exp",
            "skew",
        ], f'Contrast function can be either "logcosh", "exp" or "skew": the provided one was {g_func}.'
        assert conv_th > 0, "Convergence threshold must be positive."
        assert -1 < sil_th < 1, "SIL threshold must be in the ]-1, 1[ range."
        assert max_iter > 0, "The maximum n. of iterations must be positive."
        assert sorting_strategy is None or sorting_strategy in [
            "firing-rate",
            "neg-entropy"], f'The sorting strategy can be either "firing-rate", "neg-entropy" or None.'

        # External parameters
        self._max_sources = max_sources
        self._samp_freq = samp_freq
        self._f_e = f_e
        self._reg_factor = reg_factor
        self._g_func = g_func
        self._conv_th = conv_th
        self._sil_th = sil_th
        self._max_iter = max_iter
        self._min_spike_distance = min_spike_distance
        self._min_perc = min_perc
        self._sorting_strategy = sorting_strategy
        self._momentum = momentum
        self._prng = np.random.default_rng(seed)

        # Internal parameters to keep track of
        self._params = DecompositionParameters()
        self._is_calibrated = False  # state of the object (calibrated/not calibrated)
        self._n_mu = 0  # number of extracted MUs

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def n_mu(self) -> int:
        return self._n_mu

    def calibrate(
            self,
            emg: np.ndarray,
            min_spike_pps: float = 5,
            max_spike_pps: float = 250
    ) -> pd.DataFrame:
        """Calibrate instance on the given sEMG signal and return the extracted firings.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).
        min_spike_pps: float, default=5
            Minimum pulses-per-second (pps) for considering a MU valid.
        max_spike_pps: float, default=250
            Maximum pulses-per-second (pps) for considering a MU valid.

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        assert not self._is_calibrated, "Instance already calibrated."

        start = time.time()

        # 1. Preprocessing: extension, centering and whitening
        emg_white = self._preprocessing(emg)

        # 2. Decomposition: FastICA and source improvement
        self._params.sep_mtx = np.zeros(shape=(0, emg_white.shape[0]))  # initialize separation matrix
        self._params.spike_th = np.zeros(shape=(0,))  # initialize spike thresholds
        self._conv_bss(emg_white)

        # 3. Postprocessing: removal of inactive MUs and replicas
        min_n_spikes = int(min_spike_pps * emg.shape[1] / self._samp_freq)
        max_n_spikes = int(max_spike_pps * emg.shape[1] / self._samp_freq)
        firings = self._post_processing(emg_white, min_n_spikes, max_n_spikes)

        # Keep track of the number of MUs extracted
        self._n_mu = self._params.sep_mtx.shape[0]

        # Set instance to trained
        self._is_calibrated = True

        elapsed = time.time() - start
        logging.info(
            f"Decomposition performed in "
            f"{int(elapsed / 60):02d}min {int(elapsed / 3600):02d}s {int(elapsed % 3600):03d}ms."
        )

        return firings

    def recalibrate(
            self,
            emg: np.ndarray,
            min_spike_pps: float = 5,
            max_spike_pps: float = 250
    ) -> pd.DataFrame:
        """Recalibrate decomposition model on a new signal and return the new firings.
        
        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).
        min_spike_pps: float, default=5
            Minimum pulses-per-second (pps) for considering a MU valid.
        max_spike_pps: float, default=250
            Maximum pulses-per-second (pps) for considering a MU valid.

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        start = time.time()

        # 1. Preprocessing: extension, centering and whitening with refinement
        emg_white = self._preprocessing(emg, refine=True)

        # 2. Decomposition: FastICA and source improvement
        self._conv_bss(emg_white, from_source_idx=self._n_mu)

        # 3. Postprocessing: removal of inactive MUs and replicas
        min_n_spikes = int(min_spike_pps * emg.shape[1] / self._samp_freq)
        max_n_spikes = int(max_spike_pps * emg.shape[1] / self._samp_freq)
        firings = self._post_processing(
            emg_white,
            min_n_spikes,
            max_n_spikes,
            from_source_idx=self._n_mu
        )

        # Keep track of the number of MUs extracted
        self._n_mu = self._params.sep_mtx.shape[0]

        # Set instance to trained
        self._is_calibrated = True

        elapsed = time.time() - start
        logging.info(
            f"Decomposition performed in "
            f"{int(elapsed / 60):02d}min {int(elapsed / 3600):02d}s {int(elapsed % 3600):03d}ms."
        )

        return firings

    def decompose(
            self, emg: np.ndarray,
            min_spike_pps: float = 5,
            max_spike_pps: float = 250
    ) -> pd.DataFrame:
        """Decompose given data using pre-computed parameters.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).
        min_spike_pps: float, default=5
            Minimum pulses-per-second (pps) for considering a MU valid.
        max_spike_pps: float, default=250
            Maximum pulses-per-second (pps) for considering a MU valid.

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        assert self._is_calibrated, "The instance must be calibrated first."

        # 1. Preprocessing: extension, centering and whitening (with pre-computed matrix)
        emg_white = self._preprocessing_precomp(emg)

        # 2. Postprocessing: removal of inactive MUs and replicas
        min_n_spikes = int(min_spike_pps * emg.shape[1] / self._samp_freq)
        max_n_spikes = int(max_spike_pps * emg.shape[1] / self._samp_freq)
        firings = self._post_processing_precomp(emg_white, min_n_spikes, max_n_spikes)

        return firings

    def project(self, emg: np.ndarray) -> np.ndarray:
        """Given a raw EMG signal, compute its sources.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).

        Returns
        -------
        sources: np.ndarray
            Sources with shape (n_mu, n_samples).
        """
        assert self._is_calibrated, "The instance must be calibrated first."

        emg_white = self._preprocessing_precomp(emg)
        return self._params.sep_mtx @ emg_white

    def compute_negentropy(self, sources: np.ndarray) -> np.ndarray:
        """Given the sources, compute their neg-entropy.

        Parameters
        ----------
        sources: np.ndarray
            Source signals with shape (n_mu, n_samples).

        Returns
        -------
        negentropy: np.ndarray
            Neg-entropy value for each source.
        """
        g = g_dict[self._g_func]
        g_sources, _, _ = g(sources)
        g_std, _, _ = g(self._prng.standard_normal(size=sources.shape))
        return np.square(np.mean(g_sources, axis=1) - np.mean(g_std, axis=1))

    def reset(self) -> None:
        """Reset the internal state of the EMGSeparator."""
        self._is_calibrated = False
        self._params = DecompositionParameters()

    def save_to_file(self, filename: str) -> None:
        """Save instance to a .pkl file using pickle.
        
        Parameters
        ----------
        filename: str
            Path to the .pkl file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename: str) -> EMGSeparator:
        """Load instance from a .pkl file using pickle.
        
        Parameters
        ----------
        filename: str
            Path to the .pkl file.
        
        Returns
        -------
        obj: EMGSeparator
            Instance of EMGSeparator.
        """
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def _preprocessing(self, emg: np.ndarray, refine: bool = False) -> np.ndarray:
        """Preprocess raw sEMG signal.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).
        refine: bool, default=False
            Whether to refine the parameters or not.

        Returns
        -------
        emg_white: np.ndarray
            Preprocessed sEMG signal with shape (f_e * n_channels, n_samples).
        """
        # 1. Extension
        emg_ext = extend_signal(emg, self._f_e)
        
        if refine:
            # 2. Centering
            _, mean_vec = center_signal(emg_ext)
            self._params.mean_vec = (1 - self._momentum) * self._params.mean_vec + self._momentum * mean_vec
            emg_center = emg_ext - self._params.mean_vec

            # 3. Whitening
            _, white_mtx = whiten_signal(emg_center, self._reg_factor)
            self._params.white_mtx = (1 - self._momentum) * self._params.white_mtx + self._momentum * white_mtx
            emg_white = self._params.white_mtx @ emg_center
            
            logging.info("Mean vector and whitening matrix refined.")
        else:
            # 2. Centering
            emg_center, mean_vec = center_signal(emg_ext)
            self._params.mean_vec = mean_vec

            # 3. Whitening
            emg_white, white_mtx = whiten_signal(emg_center, self._reg_factor)
            self._params.white_mtx = white_mtx

            logging.info("Mean vector and whitening matrix computed.")

        return emg_white

    def _preprocessing_precomp(self, emg: np.ndarray) -> np.ndarray:
        """Preprocess raw sEMG signal using pre-computed parameters.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).

        Returns
        -------
        emg_white: np.ndarray
            Preprocessed sEMG signal with shape (f_e * n_channels, n_samples).
        """
        # 1. Extension
        emg_ext = extend_signal(emg, self._f_e)
        
        # 2. Centering
        emg_center = emg_ext - self._params.mean_vec
        
        # 3. Whitening
        emg_white = self._params.white_mtx @ emg_center

        logging.info("Using pre-computed mean vector and whitening matrix.")

        return emg_white

    def _conv_bss(self, emg_white: np.ndarray, from_source_idx: int = 0) -> None:
        """Perform FastICA + source improvement.

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        from_source_idx: int, default=0
            Index of the initial source to estimate (useful for recalibration).
        """
        g = g_dict[self._g_func]
        # Initialize separation vector indices
        wi_init_idx = self._init_indices(emg_white)
        for i in range(from_source_idx, self._max_sources):
            logging.info(f"----- SOURCE {i + 1} -----")

            # Initialize separation vector
            wi_init = None
            fraction_peaks = 0.75
            if i < fraction_peaks * self._max_sources:
                wi_idx = self._prng.choice(wi_init_idx, size=1)[0]
                logging.info(
                    f"Initialization done using index {wi_idx} with "
                    f"value {np.square(emg_white[:, wi_idx]).sum(axis=0):.3e}."
                )
                wi_init = emg_white[:, wi_idx]

            # 1. FastICA for the i-th unit
            wi, converged = self._fast_ica_def(emg_white, g, wi_init)
            if not converged:
                logging.info("FastICA didn't converge, reinitializing...")
                continue

            # 2. Source improvement
            wi = self._source_improvement(emg_white, wi)

            # 3. Check SIL
            sil = self._compute_sil(si=np.dot(wi, emg_white))
            if sil > self._sil_th:
                # Add wi to separation matrix
                self._params.sep_mtx = np.concatenate(
                    [self._params.sep_mtx, wi.reshape(1, -1)]
                )
                logging.info(
                    f"Source accepted (SIL = {sil:.3f}), adding vector to separation matrix, "
                    f"new shape: {self._params.sep_mtx.shape}."
                )
            else:
                logging.info(
                    f"SIL below threshold (SIL = {sil:.3f}), skipping source..."
                )

        logging.info(f"Extracted {self._params.sep_mtx.shape[0]} MUs before post-processing.")

    def _init_indices(self, emg_white: np.ndarray) -> np.ndarray:
        """Get initial estimation for separation vectors.

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).

        Returns
        -------
        wi_init_idx: np.ndarray
            Indices of the whitened data to use as initial estimation of separation vectors.
        """
        emg_sq = np.square(emg_white).sum(axis=0)
        # Consider 10ms on either side for finding peaks
        peaks, _ = find_peaks(emg_sq, distance=int(round(10e-3 * self._samp_freq)))
        peak_heights = emg_sq[peaks]
        sorted_peaks_idx = np.argsort(peak_heights)[::-1]
        # Find peaks in the whitened data to use as initialization points for the fixed-point algorithm
        max_wi_indices = peaks[sorted_peaks_idx]

        # Initialize according to a random peak in the top 25%
        top_max_wi_indices = len(max_wi_indices) // 4
        if top_max_wi_indices < 4 * self._max_sources:
            top_max_wi_indices = 4 * self._max_sources
        return max_wi_indices[:top_max_wi_indices]

    def _fast_ica_def(
            self,
            emg_white: np.ndarray,
            g: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
            wi_init: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """FastICA (deflation).

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        g: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
            Contrast function.
        wi_init: np.ndarray
            Initial separation vector.

        Returns
        -------
        wi: np.ndarray
            Separation vector with shape (n_channels,).
        converged: bool
            Whether FastICA converged or not.
        """
        n_channels = emg_white.shape[0]

        # Initialize separation vector
        if wi_init is not None:
            wi = wi_init
        else:
            wi = self._prng.standard_normal(size=(n_channels,), dtype=float)

        # De-correlate and normalize
        wi -= np.dot(self._params.sep_mtx.T @ self._params.sep_mtx, wi)
        wi /= np.linalg.norm(wi)

        # Iterate until convergence or max_iter are reached
        iter_idx = 0
        converged = False
        while iter_idx < self._max_iter:
            wi_new = _fast_ica_iter(emg_white, wi, g)
            # De-correlate and normalize
            wi_new -= np.dot(self._params.sep_mtx.T @ self._params.sep_mtx, wi_new)
            wi_new /= np.linalg.norm(wi_new)

            # Compute distance
            distance = 1 - abs(np.dot(wi_new, wi).item())
            # logging.info(f"FastICA iteration {iter_idx}: {distance:.3e}.")
            # Update separation vector
            wi = wi_new
            # Update iteration count
            iter_idx += 1

            # Check convergence
            if distance < self._conv_th:
                converged = True
                logging.info(
                    f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break

        return wi, converged

    def _source_improvement(self, emg_white: np.ndarray, wi: np.ndarray) -> np.ndarray:
        """Source improvement iteration.

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        wi: np.ndarray
            Separation vector with shape (n_channels,).

        Returns
        -------
        wi: np.ndarray
            Refined separation vector with shape (n_channels,).
        """
        cov = np.inf
        iter_idx = 0
        while iter_idx < self._max_iter:
            # Detect spikes
            spike_loc, cov_new, _ = self._detect_spikes(
                si=np.dot(wi, emg_white), compute_cov=True
            )

            # Compute new separation vector, apply de-correlation and normalize it
            wi_new = emg_white[:, spike_loc].mean(axis=1)
            wi_new -= np.dot(self._params.sep_mtx.T @ self._params.sep_mtx, wi_new)
            wi_new /= np.linalg.norm(wi_new)

            # Check CoV-ISI
            if cov_new is None:
                logging.info(f"Spike detection at step {iter_idx} failed, aborting...")
                break
            if abs(cov_new - cov) < self._conv_th:
                logging.info(
                    f"Source improvement converged at step {iter_idx}, "
                    f"the distance is {abs(cov_new - cov):.3e}."
                )
                wi = wi_new
                break
            if cov_new > cov:
                logging.info(
                    f"Spike detection at step {iter_idx} resulted in an "
                    f"increase from {cov} to {cov_new}, aborting..."
                )
                break

            logging.info(
                f"Spike detection at step {iter_idx} resulted in a "
                f"decrease from {cov} to {cov_new}."
            )
            wi = wi_new
            cov = cov_new
            iter_idx += 1

        return wi

    def _detect_spikes(
            self, si: np.ndarray, compute_cov: bool = False, threshold: float | None = None
    ) -> tuple[np.ndarray, float | None, float | None]:
        """Detect spikes in the given source.

        Parameters
        ----------
        si: np.ndarray
            Estimated source with shape (n_samples,).
        compute_cov: bool, default=False
            Whether to compute the Coefficient of Variation (CoV) or not.
        threshold: float | None, default=None
            Threshold for spike/noise classification.

        Returns
        -------
        wi: np.ndarray
            Refined separation vector with shape (n_channels,).
        cov: float | None
            Coefficient of Variation of the Inter-Spike Interval.
        threshold: float | None
            Threshold for spike/noise classification.
        """
        # Find peaks of squared source
        si_sq = np.square(si)
        peak_distance = int(round((self._min_spike_distance / 1000.0) * self._samp_freq))
        peaks, _ = find_peaks(si_sq, distance=peak_distance)
        sq_peaks = si_sq[peaks]

        # Use threshold to identify spikes, if provided
        if threshold is not None:
            logging.info(f"Using pre-computed threshold {threshold}.")
            spike_loc = peaks[sq_peaks > threshold]
            return spike_loc, None, None

        # Perform k-means with 2 clusters (high and small peaks)
        centroids, labels = kmeans2(
            sq_peaks.reshape(-1, 1), k=2, minit="++", seed=self._prng
        )
        # Consider only high peaks (i.e. cluster with highest centroid)
        high_cluster_idx = np.argmax(centroids)
        spike_loc = peaks[labels == high_cluster_idx]

        # Compute threshold
        new_threshold = centroids.mean()

        # Compute metric
        if compute_cov:  # CoV-ISI
            isi = np.diff(spike_loc)

            if isi.size == 0 or isi.mean() == 0:
                logging.info("Only single peak detected, aborting...")
                return spike_loc, None, new_threshold

            cov = np.std(isi) / np.mean(isi)
            logging.info(
                f"{spike_loc.size}/{peaks.size} peaks considered high [CoV-ISI = {cov:.3f}]"
            )
            return spike_loc, cov, new_threshold

        return spike_loc, None, new_threshold

    def _compute_sil(
            self, si: np.ndarray
    ) -> float:
        """Compute the SIL measure for the given source.

        Parameters
        ----------
        si: np.ndarray
            Estimated source with shape (n_samples,).

        Returns
        -------
        sil: float
            SIL measure.
        """
        # Perform k-means on squared source
        si_sq = np.square(si)
        peak_distance = int(round((self._min_spike_distance / 1000.0) * self._samp_freq))
        peaks, _ = find_peaks(si_sq, distance=peak_distance)
        sq_peaks = si_sq[peaks]
        centroids, labels = kmeans2(
            sq_peaks.reshape(-1, 1), k=2, minit="++", seed=self._prng
        )

        # Compute SIL
        within_sum, between_sum = 0, 0
        for i in range(sq_peaks.size):
            within_sum += abs(sq_peaks[i] - centroids[labels[i], 0])
            between_sum += abs(sq_peaks[i] - centroids[1 - labels[i], 0])

        sil = (between_sum - within_sum) / max(within_sum, between_sum)
        return sil

    def _post_processing(
            self, emg_white: np.ndarray, min_n_spikes: int, max_n_spikes: int, from_source_idx: int = 0
    ) -> pd.DataFrame:
        """Post-process the detected spikes by removing replicas and inactive MUs (train mode).

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        min_n_spikes: int
            Minimum number of spikes to consider a MU as active.
        max_n_spikes: int
            Maximum number of spikes to consider a MU as active.
        from_source_idx: int, default=0
            Index of the initial source to estimate (useful for recalibration).

        Returns
        -------
        firings_list: pd.DataFrame
            A DataFrame with the firing times of every MU.
        """
        # Step 1: obtain firing times and remove inactive MUs
        _, n_samples = emg_white.shape

        # Compute sources
        sources = self._params.sep_mtx @ emg_white
        n_sources, _ = sources.shape

        firings_dict: dict[int, np.ndarray] = {}
        invalid_mus: list[int] = []
        valid_count = 0
        for i in range(n_sources):
            # Detect spikes
            spike_loc, _, spike_th = self._detect_spikes(
                si=sources[i]  # , threshold=self._params.spike_th[i]
            )
            if i < from_source_idx:  # recalibration
                firings_dict[valid_count] = spike_loc / self._samp_freq
                valid_count += 1
                # Refine threshold
                new_spike_th = (1 - self._momentum) * self._params.spike_th[i] + self._momentum * spike_th
                logging.info(
                    f"The {i}-th MU fired {spike_loc.size} times and was detected in the previous calibration, "
                    f"refining threshold from {self._params.spike_th[i]} to {new_spike_th}."
                )
                self._params.spike_th[i] = new_spike_th
            else:  # first calibration
                if min_n_spikes <= spike_loc.size <= max_n_spikes:  # save firing time
                    firings_dict[valid_count] = spike_loc / self._samp_freq
                    valid_count += 1
                    # Save threshold
                    self._params.spike_th = np.concatenate([self._params.spike_th, (spike_th,)])
                    logging.info(
                        f"Saving threshold {spike_th} for {i}-th MU."
                    )
                else:
                    invalid_mus.append(i)  # set i-th MU as invalid
                    logging.info(
                        f"The {i}-th MU fired {spike_loc.size} times "
                        f"while valid range was [{min_n_spikes}, {max_n_spikes}], removing it..."
                    )

        # Remove invalid MUs
        self._params.sep_mtx = np.delete(self._params.sep_mtx, invalid_mus, axis=0)
        sources = np.delete(sources, invalid_mus, axis=0)

        # Step 2: remove MUs delayed replicas
        mus_to_remove = self._find_duplicates(firings_dict, sources, from_source_idx)
        self._params.sep_mtx = np.delete(self._params.sep_mtx, mus_to_remove, axis=0)
        self._params.spike_th = np.delete(self._params.spike_th, mus_to_remove, axis=0)
        sources = np.delete(sources, mus_to_remove, axis=0)
        for mu in mus_to_remove:
            del firings_dict[mu]

        logging.info(f"Extracted {self._params.sep_mtx.shape[0]} MUs after replicas removal.")

        # Step 3: pack results in a Pandas DataFrame
        # Compute firing rate and neg-entropy
        firings_list: list[np.ndarray] = list(firings_dict.values())
        firing_rate = np.array(
            list(map(lambda x: x.size / n_samples * self._samp_freq, firings_list))
        )
        negentropy = self.compute_negentropy(sources)

        # Sort MUs (in descending order)
        if self._sorting_strategy is not None:
            sort_by_negentropy = self._sorting_strategy == "neg-entropy"
    
            idx = (
                np.argsort(negentropy)[::-1]
                if sort_by_negentropy
                else np.argsort(firing_rate)[::-1]
            )
            self._params.sep_mtx = self._params.sep_mtx[idx]
            self._params.spike_th = self._params.spike_th[idx]
            firings_list = [firings_list[i] for i in idx]
            firing_rate = firing_rate[idx]
            negentropy = negentropy[idx]

        firings = pd.DataFrame([
            {"MU index": i, "Firing time": f_time, "Firing rate": f_rate, "Neg-entropy": neg}
            for i, (f_times, f_rate, neg) in enumerate(zip(firings_list, firing_rate, negentropy))
            for f_time in f_times
        ])

        return firings

    def _post_processing_precomp(
            self, emg_white: np.ndarray, min_n_spikes: int, max_n_spikes: int
    ) -> pd.DataFrame:
        """Post-process the detected spikes using pre-computed parameters.

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        min_n_spikes: int
            Minimum number of spikes to consider a MU as active.
        max_n_spikes: int
            Maximum number of spikes to consider a MU as active.

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        _, n_samples = emg_white.shape

        # Compute sources
        sources = self._params.sep_mtx @ emg_white
        n_sources, _ = sources.shape
        # Compute neg-entropy
        neg_entropy = self.compute_negentropy(sources)

        firings_tmp: list[dict[str, float]] = []
        for i in range(n_sources):
            # Detect spikes with pre-computed threshold
            spike_loc, _, _ = self._detect_spikes(
                si=sources[i], threshold=self._params.spike_th[i]
            )
            if min_n_spikes <= spike_loc.size <= max_n_spikes:
                f_rate = spike_loc.size / n_samples * self._samp_freq
                firings_tmp.extend(
                    [
                        {
                            "MU index": i,
                            "Firing time": s / self._samp_freq,
                            "Firing rate": f_rate,
                            "Neg-entropy": neg_entropy[i],
                        }
                        for s in spike_loc
                    ]
                )
            else:
                logging.info(
                    f"The {i}-th MU fired {spike_loc.size} times while "
                    f"valid range was [{min_n_spikes}, {max_n_spikes}], removing it..."
                )
        # Convert to Pandas DataFrame
        firings = pd.DataFrame(firings_tmp)

        return firings
    
    def _find_duplicates(
        self,
        firings_dict: dict[int, np.ndarray],
        sources: np.ndarray,
        from_source_idx: int
    ) -> list[int]:
        """Find duplicate MUs.
        
        Parameters
        ----------
        firings_dict: dict[int, np.ndarray]
            Dictionary containing an array with discharge timings for each MU.
        sources: np.ndarray
            Source signal with shape (n_mu, n_samples).
        from_source_idx: int
            Index of the initial source to consider (useful to avoid removing MUs extracted in previous sessions).

        Returns
        -------
        mus_to_remove: list[int]
            List containing the duplicate MUs to remove.
        """
        cur_mu = 0
        mu_idx = list(firings_dict.keys())
        duplicate_mus: dict[int, list[int]] = {}
        while cur_mu < len(mu_idx):
            # Find index of replicas by checking synchronization
            i = 1
            while i < len(mu_idx) - cur_mu:
                in_sync1 = self._sync_correlation(
                    firings_dict[mu_idx[cur_mu]],
                    firings_dict[mu_idx[cur_mu + i]],
                    time_lock=0.001,  # = 1 ms
                )
                in_sync2 = self._sync_correlation(
                    firings_dict[mu_idx[cur_mu + i]],
                    firings_dict[mu_idx[cur_mu]],
                    time_lock=0.001,  # = 1 ms
                )
                # Since the number of spikes in the two MUs may differ, check if both 
                # share more than min_perc synchronized spikes
                if in_sync1 and in_sync2:
                    duplicate_mus[mu_idx[cur_mu]] = duplicate_mus.get(
                        mu_idx[cur_mu], []
                    ) + [mu_idx[cur_mu + i]]
                    del mu_idx[cur_mu + i]
                else:
                    i += 1
            cur_mu += 1

        # Decide which duplicates to remove
        mus_to_remove: list[int] = []
        for main_mu, dup_mus in duplicate_mus.items():
            # Unify duplicate MUs
            dup_mus = [main_mu] + dup_mus
            dup_str = ", ".join([f"{mu}" for mu in dup_mus])
            logging.info(f"Found group of duplicate MUs: {dup_str}.")

            # Check if the duplicate group contains MUs extracted on previous session
            dup_mus = list(filter(lambda mu: mu >= from_source_idx, dup_mus))
            if len(dup_mus) == 0:
                logging.info(f"Skipping since the detected duplicates MUs were extracted on previous session.")
                continue

            # Compute SIL for every MU, and keep only the one with the highest SIL
            sil_scores = [
                (dup_mu, self._compute_sil(si=sources[dup_mu])) for dup_mu in dup_mus
            ]
            mu_keep = max(sil_scores, key=lambda t: t[1])
            logging.info(f"Keeping MU {mu_keep[0]} (SIL = {mu_keep[1]:.3f}).")

            # Mark duplicates
            dup_mus.remove(mu_keep[0])
            mus_to_remove.extend(dup_mus)
        
        return mus_to_remove

    def _sync_correlation(
        self,
        firings_ref: np.ndarray,
        firings_sec: np.ndarray,
        time_lock: float,
        win_len: float = 0.01
    ) -> bool:
        """Check if two MUAPTs are in sync.

        Parameters
        ----------
        firings_ref: np.ndarray
            Firing times for reference MU.
        firings_sec: np.ndarray
            Firing times for secondary MU.
        time_lock: float
            Maximum number of seconds between two spikes to consider them time-locked.
        win_len: float, default=0.01
            Length of the window for sync calculation (in seconds).

        Returns
        -------
        in_sync: bool
            Whether the two MUAPTs are in sync or not.
        """
        sync: list[float] = []
        for firing_ref in firings_ref:
            fire_interval = firings_sec - firing_ref
            idx = np.flatnonzero(
                (fire_interval >= -win_len) & (fire_interval <= win_len)
            )
            if idx.size != 0:
                sync.extend(fire_interval[idx])

        # Compute histogram of relative timing
        hist, bin_edges = np.histogram(sync)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        max_bin = hist.argmax()
        common_idx = np.flatnonzero(
            (sync >= bin_centers[max_bin] - time_lock) & (sync <= bin_centers[max_bin] + time_lock)
        )

        return common_idx.size / firings_ref.size > self._min_perc
