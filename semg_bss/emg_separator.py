from __future__ import annotations

import logging
import pickle
import time

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
    gx_sec = alpha * (1 - gx_prime**2)

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

    gx = -np.exp(-(x**2) / 2)
    # Compute G

    # Compute G'
    gx_prime = -x * gx
    # Compute G''
    gx_sec = (x**2 - 1) * gx

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
    gx = x**3 / 3
    # Compute G'
    gx_prime = x**2
    # Compute G''
    gx_sec = 2 * x

    return gx, gx_prime, gx_sec


class EMGSeparator:
    """Perform blind source separation of sEMG signals via convolutive FastICA + source improvement.

    Parameters
    ----------
    max_comp: int
        Maximum n. of components that can be extracted.
    fs: float
        Sampling frequency of the signal.
    f_e: int, default=0
        Extension factor for the signal.
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
    min_perc: float, default=0.3
        Minimum percentage of common firings for considering two MUs as duplicates.
    seed: Optional[int], default=None
        Seed for the internal PRNG.

    Attributes
    ----------
    _is_calibrated: bool
        Whether the instance is calibrated or not.
    _max_comp: int
        Maximum n. of components that can be extracted.
    _f_e: int, default=0
        Extension factor for the signal.
    _g: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Contrast function for FastICA.
    _g_name: str
        Name of the contrast function (for pickling/unpickling).
    _conv_th: float, default=1e-4
        Threshold for convergence.
    _sil_th: float, default=0.9
        Threshold for source acceptance based on the silhouette score.
    _max_iter: int, default=100
        Maximum n. of iterations.
    _min_spike_distance: float, default=10
        Minimum distance between two spikes.
    _min_perc: float, default=0.3
        Minimum percentage of common firings for considering two MUs as duplicates.
    _prng: np.Generator, default=None
        Actual PRNG.
    _white_mtx: np.ndarray | None
        Whitening matrix.
    _sep_mtx: np.ndarray | None
        Separation matrix.
    _spike_th: np.ndarray
        Threshold for spike/noise classification for each MU.
    _n_mu: int
        Number of extracted MUs.
    """

    def __init__(
        self,
        max_comp: int,
        fs: float,
        f_e: int = 0,
        g_func: str = "logcosh",
        conv_th: float = 1e-4,
        sil_th: float = 0.9,
        max_iter: int = 100,
        min_spike_distance: float = 10,
        min_perc: float = 0.3,
        seed: int | None = None,
    ):
        # Dictionary for contrast functions
        g_dict = {"logcosh": _logcosh, "exp": _exp, "skew": _skew}

        # Parameter check
        assert max_comp > 0, "The maximum n. of components must be positive."
        assert g_func in [
            "logcosh",
            "exp",
            "skew",
        ], f'Contrast function can be either "logcosh", "exp" or "skew": the provided one was {g_func}.'
        assert conv_th > 0, "Convergence threshold must be positive."
        assert -1 < sil_th < 1, "SIL threshold must be in the ]-1, 1[ range."
        assert max_iter > 0, "The maximum n. of iterations must be positive."

        # External parameters
        self._max_comp = max_comp
        self._fs = fs
        self._f_e = f_e
        self._g = g_dict[g_func]
        self._g_name = g_func
        self._conv_th = conv_th
        self._sil_th = sil_th
        self._max_iter = max_iter
        self._min_spike_distance = min_spike_distance
        self._min_perc = min_perc
        self._prng = np.random.default_rng(seed)

        # Internal parameters to keep track of
        self._is_calibrated = False  # state of the object (calibrated/not calibrated)
        self._white_mtx = None  # whitening matrix
        self._sep_mtx = None  # separation matrix
        self._spike_th = np.zeros(
            shape=(0,), dtype=float
        )  # threshold for spike/noise classification
        self._n_mu = 0  # number of extracted MUs
        self._sort_by_negentropy = False  # sorting strategy
    
    def __get_state__(self):
        d = self.__dict__.copy()
        if "_g" in d:
            del d["_g"]
        return d
    
    def __set_state__(self, d):
        # Dictionary for contrast functions
        g_dict = {"logcosh": _logcosh, "exp": _exp, "skew": _skew}

        if "_g" in d and "_g_name" in d:
            d["_g"] = g_dict[d["_g_name"]]
        self.__dict__.update(d)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def n_mu(self) -> int:
        return self._n_mu

    def calibrate(
        self,
        emg: np.ndarray,
        min_spike_pps: float = 0.5,
        sort_by_negentropy: bool = False
    ) -> pd.DataFrame:
        """Calibrate instance on the given sEMG signal and return the extracted firings.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).
        min_spike_pps: float, default=1
            Minimum pulses-per-second (pps) for considering a MU valid.
        sort_by_negentropy: bool, default=False
            Whether to sort MUs by neg-entropy or firing rate (default).

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        assert not self._is_calibrated, "Instance already calibrated."
        
        start = time.time()

        self._sort_by_negentropy = sort_by_negentropy

        # 1. Preprocessing: extension, centering and whitening
        emg_white = self._preprocessing_train(emg)

        # 2. Decomposition: FastICA and source improvement
        self._sep_mtx = np.zeros(shape=(emg_white.shape[0], 0))  # initialize separation matrix
        self._decomposition(emg_white)

        # 3. Postprocessing: removal of inactive MUs and replicas
        min_n_spikes = int(min_spike_pps * emg.shape[1] / self._fs)
        firings = self._post_processing_train(emg_white, min_n_spikes)

        # Keep track of the number of MUs extracted
        self._n_mu = self._sep_mtx.shape[1]

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
        min_spike_pps: float = 1
    ) -> pd.DataFrame:
        """Recalibrate decomposition model on a new signal and return the new firings.
        
        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).
        min_spike_pps: float, default=1
            Minimum pulses-per-second (pps) for considering a MU valid.

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        start = time.time()

        # 1. Preprocessing: extension, centering and whitening
        emg_white = self._preprocessing_train(emg)  # refine whitening matrix

        # 2. Decomposition: FastICA and source improvement
        self._decomposition(emg_white, from_source_idx=self._n_mu)

        # 3. Postprocessing: removal of inactive MUs and replicas
        min_n_spikes = int(min_spike_pps * emg.shape[1] / self._fs)
        firings = self._post_processing_train(
            emg_white,
            min_n_spikes,
            from_source_idx=self._n_mu
        )

        # Keep track of the number of MUs extracted
        self._n_mu = self._sep_mtx.shape[1]

        # Set instance to trained
        self._is_calibrated = True

        elapsed = time.time() - start
        logging.info(
            f"Decomposition performed in "
            f"{int(elapsed / 60):02d}min {int(elapsed / 3600):02d}s {int(elapsed % 3600):03d}ms."
        )

        return firings

    def transform(
        self, emg: np.ndarray,
        min_spike_pps: float = 1
    ) -> pd.DataFrame:
        """Decompose given data using pre-computed parameters.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).
        min_spike_pps: float, default=1
            Minimum pulses-per-second (pps) for considering a MU valid.

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        assert self._is_calibrated, "The instance must be calibrated first."

        # 1. Preprocessing: extension, centering and whitening (with pre-computed matrix)
        emg_white = self._preprocessing_inference(emg)

        # 2. Postprocessing: removal of inactive MUs and replicas
        min_n_spikes = int(min_spike_pps * emg.shape[1] / self._fs)
        firings = self._post_processing_inference(emg_white, min_n_spikes)

        return firings

    def project(self, emg: np.ndarray) -> np.ndarray:
        """Given a raw EMG signal, compute the source MUAPTs.

        Parameters
        ----------
        emg: np.ndarray
            Raw sEMG signal with shape (n_channels, n_samples).

        Returns
        -------
        muapts: np.ndarray
            Source MUAPTs with shape (n_mu, n_samples).
        """
        assert self._is_calibrated, "The instance must be calibrated first."

        emg_white = self._preprocessing_inference(emg)
        return self._sep_mtx.T @ emg_white

    def compute_negentropy(self, muapts) -> np.ndarray:
        """Given a set of MUAPTs, compute their neg-entropy.

        Parameters
        ----------
        muapts: np.ndarray
            Source MUAPTs with shape (n_mu, n_samples).

        Returns
        -------
        negentropy: np.ndarray
            Neg-entropy value for each MU.
        """
        g_muapts, _, _ = self._g(muapts)
        g_std, _, _ = self._g(self._prng.standard_normal(size=muapts.shape))
        return np.square(np.mean(g_muapts, axis=1) - np.mean(g_std, axis=1))

    def reset(self) -> None:
        """Reset the internal state of the EMGSeparator."""
        self._is_calibrated = False
        self._white_mtx = None
        self._sep_mtx = None
        self._spike_th = np.zeros(shape=(0,), dtype=float)
        self._n_mu = None
    
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

    def _preprocessing_train(self, emg: np.ndarray) -> np.ndarray:
        """Preprocess raw sEMG signal (train mode).

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
        emg_center = center_signal(emg_ext)
        # 3. Whitening
        if self._white_mtx is None:  # first calibration
            emg_white, self._white_mtx = whiten_signal(emg_center)
            logging.info("Whitening matrix computed.")
        else:  # recalibration
            emg_white, white_mtx = whiten_signal(emg_center)
            self._white_mtx = (self._white_mtx + white_mtx) / 2  # take mean
            logging.info("Whitening matrix refined.")

        return emg_white

    def _preprocessing_inference(self, emg: np.ndarray) -> np.ndarray:
        """Preprocess raw sEMG signal (inference mode).

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
        emg_center = center_signal(emg_ext)
        # 3. Whitening
        logging.info("Using pre-computed whitening matrix.")
        emg_white = self._white_mtx @ emg_center

        return emg_white

    def _decomposition(self, emg_white: np.ndarray, from_source_idx: int = 0) -> None:
        """Perform FastICA + source improvement.

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        from_source_idx: int, default=0
            Index of the initial source to estimate (useful for recalibration).
        """
        # Initialize separation vector indices
        wi_init_idx = self._init_indices(emg_white)
        for i in range(from_source_idx, self._max_comp):
            logging.info(f"----- SOURCE {i + 1} -----")

            # Initialize separation vector
            wi_init = None
            fraction_peaks = 0.75
            if i < fraction_peaks * self._max_comp:
                wi_idx = self._prng.choice(wi_init_idx, size=1)[0]
                logging.info(
                    f"Initialization done using index {wi_idx} with "
                    f"value {np.square(emg_white[:, wi_idx]).sum(axis=0):.3e}."
                )
                wi_init = emg_white[:, wi_idx]

            # 1. FastICA for the i-th unit
            wi, converged = self._fast_ica_one_unit(emg_white, wi_init)
            if not converged:
                logging.info("FastICA didn't converge, reinitializing...")
                continue

            # 2. Source improvement
            wi = self._source_improvement(emg_white, wi)

            # 3. Check SIL
            sil, th = self._compute_sil(si=np.dot(wi, emg_white), return_threshold=True)
            if sil >= self._sil_th:
                # Add wi to separation matrix
                self._sep_mtx = np.concatenate(
                    [self._sep_mtx, wi.reshape(-1, 1)], axis=1
                )
                self._spike_th = np.concatenate([self._spike_th, (th,)], axis=0)
                logging.info(
                    f"Source accepted (SIL = {sil:.3f}), adding vector to separation matrix, "
                    f"new shape: {self._sep_mtx.shape}."
                )
            else:
                logging.info(
                    f"SIL below threshold (SIL = {sil:.3f}), skipping source..."
                )

        logging.info(f"Extracted {self._sep_mtx.shape[1]} MUs before post-processing.")

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
        peaks, _ = find_peaks(emg_sq, distance=int(round(10e-3 * self._fs)))
        peak_heights = emg_sq[peaks]
        sorted_peaks_idx = np.argsort(peak_heights)[::-1]
        # Find peaks in the whitened data to use as initialization points for the fixed-point algorithm
        max_wi_indices = peaks[sorted_peaks_idx]

        # Initialize according to a random peak in the top 25%
        top_max_wi_indices = len(max_wi_indices) // 4
        if top_max_wi_indices < 4 * self._max_comp:
            top_max_wi_indices = 4 * self._max_comp
        return max_wi_indices[:top_max_wi_indices]

    def _fast_ica_one_unit(
        self, emg_white: np.ndarray, wi_init: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """FastICA one-unit implementation.

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
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

        # Decorrelate and normalize
        wi -= np.dot(self._sep_mtx @ self._sep_mtx.T, wi)
        wi /= np.linalg.norm(wi)

        # Iterate until convergence or max_iter are reached
        iter_idx = 0
        converged = False
        while iter_idx < self._max_iter:
            # (n_channels,) @ (n_channels, n_samples) -> (n_samples,)
            _, g_ws_prime, g_ws_sec = self._g(np.dot(wi, emg_white))
            # (n_channels, n_samples) * (n_samples,) -> (n_channels, 1)
            t1 = (emg_white * g_ws_prime).mean(axis=-1)
            # E[(n_samples,)] * (n_channels, 1) -> (n_channels, 1)
            t2 = g_ws_sec.mean() * wi
            # Compute new separation vector
            wi_new = t1 - t2
            # Decorrelate and normalize
            wi_new -= np.dot(self._sep_mtx @ self._sep_mtx.T, wi_new)
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

            # Compute new separation vector, apply decorrelation and normalize it
            wi_new = emg_white[:, spike_loc].mean(axis=1)
            wi_new -= np.dot(self._sep_mtx @ self._sep_mtx.T, wi_new)
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
        peak_distance = int(round((self._min_spike_distance / 1000.0) * self._fs))
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
        self, si: np.ndarray, return_threshold: bool = False
    ) -> tuple[float, float | None]:
        """Compute the SIL measure for the given source.

        Parameters
        ----------
        si: np.ndarray
            Estimated source with shape (n_samples,).
        return_threshold: bool, default=False
            Whether to return the thresholds for kmeans or not.

        Returns
        -------
        sil: float
            SIL measure.
        threshold: float | None

        """
        # Perform k-means on squared source
        si_sq = np.square(si)
        peak_distance = int(round((self._min_spike_distance / 1000.0) * self._fs))
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

        # Compute threshold, if required
        threshold = centroids.mean() if return_threshold else None

        sil = (between_sum - within_sum) / max(within_sum, between_sum)
        return sil, threshold

    def _post_processing_train(
        self, emg_white: np.ndarray, min_n_spikes: int, from_source_idx: int = 0
    ) -> pd.DataFrame:
        """Post-process the detected spikes by removing replicas and inactive MUs (train mode).

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        min_n_spikes: int
            Minimum number of spikes to consider a MU as active.
        from_source_idx: int, default=0
            Index of the initial source to estimate (useful for recalibration).

        Returns
        -------
        firings_list: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        # Step 1: obtain firing times and remove inactive MUs
        _, n_samples = emg_white.shape

        # Compute MUAPTs waveforms
        muapts = self._sep_mtx.T @ emg_white
        n_comp, _ = muapts.shape

        firings_dict: dict[int, np.ndarray] = {}
        invalid_mus: list[int] = []
        valid_count = 0
        for i in range(n_comp):
            # Detect spikes
            spike_loc, _, _ = self._detect_spikes(
                si=muapts[i], threshold=self._spike_th[i]
            )
            if i < from_source_idx:
                # Save firing time
                firings_dict[valid_count] = spike_loc / self._fs
                valid_count += 1
                logging.info(
                    f"The {i}-th MU was detected in the previous calibration, keeping it..."
                )
            else:
                if spike_loc.size >= min_n_spikes:
                    # Save firing time
                    firings_dict[valid_count] = spike_loc / self._fs
                    valid_count += 1
                else:
                    invalid_mus.append(i)  # set i-th MU as invalid
                    logging.info(
                        f"The {i}-th MU fired only {spike_loc.size} times < {min_n_spikes}, removing it..."
                    )

        # Remove invalid MUs
        self._sep_mtx = np.delete(self._sep_mtx, invalid_mus, axis=1)
        self._spike_th = np.delete(self._spike_th, invalid_mus, axis=0)
        muapts = np.delete(muapts, invalid_mus, axis=0)

        # Step 2: remove MUs delayed replicas
        cur_mu = 0
        mu_idx = list(firings_dict.keys())
        duplicate_mus: dict[int, list[int]] = {}
        while cur_mu < len(mu_idx):
            # Find index of replicas by checking synchronization
            i = 1
            while i < len(mu_idx) - cur_mu:
                in_sync = self._sync_correlation(
                    firings_dict[mu_idx[cur_mu]],
                    firings_dict[mu_idx[cur_mu + i]],
                    win_len=0.01,  # = 10 ms
                )
                if in_sync:
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

            # Compute SIL for every MU, and keep only the one with the highest SIL
            sil_scores = [
                (dup_mu, self._compute_sil(si=muapts[dup_mu])[0]) for dup_mu in dup_mus
            ]
            mu_keep = max(sil_scores, key=lambda t: t[1])
            logging.info(f"Keeping MU {mu_keep[0]} (SIL = {mu_keep[1]:.3f}).")

            # Mark duplicates
            dup_mus.remove(mu_keep[0])
            mus_to_remove.extend(dup_mus)
        # Remove duplicates
        self._sep_mtx = np.delete(self._sep_mtx, mus_to_remove, axis=1)
        self._spike_th = np.delete(self._spike_th, mus_to_remove, axis=0)
        for mu in mus_to_remove:
            del firings_dict[mu]
        firings_list: list[np.ndarray] = list(firings_dict.values())

        logging.info(f"Extracted {self._sep_mtx.shape[1]} MUs after post-processing.")

        # Compute firing rate and neg-entropy
        firing_rate = np.array(
            list(map(lambda x: x.size / n_samples * self._fs, firings_list))
        )
        muapts = self._sep_mtx.T @ emg_white
        neg_entropy = self.compute_negentropy(muapts)

        # Sort MUs (in descending order)
        idx = (
            np.argsort(neg_entropy)[::-1]
            if self._sort_by_negentropy
            else np.argsort(firing_rate)[::-1]
        )
        self._sep_mtx = self._sep_mtx[:, idx]
        self._spike_th = self._spike_th[idx]
        firings_list = [firings_list[i] for i in idx]
        firing_rate = firing_rate[idx]
        neg_entropy = neg_entropy[idx]

        # Convert to Pandas DataFrame
        firings = pd.DataFrame([
            {"MU index": i, "Firing time": f_time, "Firing rate": f_rate, "Neg-entropy": neg}
            for i, (f_times, f_rate, neg) in enumerate(zip(firings_list, firing_rate, neg_entropy))
            for f_time in f_times
        ])

        return firings

    def _post_processing_inference(self, emg_white: np.ndarray, min_n_spikes: int) -> pd.DataFrame:
        """Post-process the detected spikes (simplified for inference mode).

        Parameters
        ----------
        emg_white: np.ndarray
            Pre-whitened sEMG signal with shape (n_channels, n_samples).
        min_n_spikes: int
            Minimum number of spikes to consider a MU as active.

        Returns
        -------
        firings: pd.DataFrame
            DataFrame with the firing times of every MU.
        """
        _, n_samples = emg_white.shape

        # Compute MUAPTs waveforms
        muapts = self._sep_mtx.T @ emg_white
        n_comp, _ = muapts.shape
        # Compute neg-entropy
        neg_entropy = self.compute_negentropy(muapts)

        firings_tmp: list[dict[str, float]] = []
        for i in range(n_comp):
            # Detect spikes with pre-computed threshold
            spike_loc, _, _ = self._detect_spikes(
                si=muapts[i], threshold=self._spike_th[i]
            )
            if spike_loc.size >= min_n_spikes:
                f_rate = spike_loc.size / n_samples * self._fs
                firings_tmp.extend(
                    [
                        {
                            "MU index": i,
                            "Firing time": s / self._fs,
                            "Firing rate": f_rate,
                            "Neg-entropy": neg_entropy[i],
                        }
                        for s in spike_loc
                    ]
                )
            else:
                logging.info(
                    f"The {i}-th MU fired only {spike_loc.size} times < {min_n_spikes}, removing it..."
                )
        # Convert to Pandas DataFrame
        firings = pd.DataFrame(firings_tmp)

        return firings

    def _sync_correlation(
        self, firings_ref: np.ndarray, firings_sec: np.ndarray, win_len: float
    ) -> bool:
        """Check if two MUAPTs are in sync.

        Parameters
        ----------
        firings_ref: np.ndarray
            Firing times for reference MU.
        firings_sec: np.ndarray
            Firing times for secondary MU.
        win_len: float
            Length of the window for sync calculation.

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
        hist, bin_edges = np.histogram(sync, bins=100)
        max_bin = hist.argmax()
        sync_idx = np.flatnonzero(
            (sync >= bin_edges[max_bin]) & (sync <= bin_edges[max_bin + 1])
        )
        common = sync_idx.size / firings_ref.size
        in_sync = common > self._min_perc

        return in_sync
