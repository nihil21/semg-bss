import numpy as np


def signal_fft(s: np.ndarray) -> np.ndarray:
    """Compute the FFT of the input signal.

    Parameters
    ----------
    s: np.ndarray
        Input signal with shape (n_channels, n_samples).

    Returns
    -------
    spectrum: np.ndarray
        Per-channel frequency spectrum.
    """
    n_channels, n_samples = s.shape
    spectrum_len = n_samples // 2
    spectrum = np.zeros(shape=(n_channels, spectrum_len), dtype=float)
    for i in range(n_channels):
        cur_spectrum = np.abs(np.fft.fft(s[i])) / n_samples
        cur_spectrum = cur_spectrum[range(spectrum_len)]
        spectrum[i] = cur_spectrum

    return spectrum
