import argparse
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearnex import patch_sklearn

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/hyser_dataset"
CACHE_DIR = "cache"

FS_EMG = 2048


def sine_wave(time: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * time + phase)


def square_wave(time: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    return amp * signal.square(2 * np.pi * freq * time + phase)


def sawtooth_wave(time: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    return amp * signal.sawtooth(2 * np.pi * freq * time + phase)


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=1, type=int, help="Subject id (in range [1, 20]")
    ap.add_argument("--session", default=1, type=int, help="Session id (in {1, 2})")
    ap.add_argument("--finger", default=1, type=int, help="Finger id (in {1, 2, 3, 4, 5})")
    ap.add_argument("--trial", default=1, type=int, help="Trial id (in {1, 2, 3})")
    ap.add_argument("--muscle", default=1, type=int, help="Muscle id (in {1, 2})")
    ap.add_argument("-r", default=4, type=int, help="Extension factor")
    ap.add_argument("--n_comp", default=300, type=int, help="Number of components to extract")
    ap.add_argument("--strategy", default="deflation", type=str, help="FastICA strategy")
    ap.add_argument("--g_func", default="logcosh", type=str, help="Non-quadratic function G")
    ap.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations")
    ap.add_argument("--ica_th", default=1e-4, type=float, help="Threshold for ICA")
    ap.add_argument("--sil_th", default=0.6, type=float, help="Threshold for SIL")
    ap.add_argument("--seed", default=None, type=int, help="Seed for PRNG.")
    ap.add_argument("--mb_size", default=100, type=int, help="Size of the mini-batch (in ms).")

    args = vars(ap.parse_args())

    # Read input arguments
    subject = args["subject"]
    session = args["session"]
    finger = args["finger"]
    trial = args["trial"]
    muscle = args["muscle"]
    r = args["r"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    ica_th = args["ica_th"]
    sil_th = args["sil_th"]
    seed = args["seed"]
    mb_size = args["mb_size"]

    # Check input
    assert subject in range(1, 21), "Subject id must be in range [1, 20]."
    assert session in [1, 2], "Session id must be in {1, 2}."
    assert finger in [1, 2, 3, 4, 5], "Finger id must be in {1, 2, 3, 4, 5}."
    assert trial in [1, 2, 3], "Sample id must be in {1, 2, 3}."
    assert muscle in [1, 2], "Muscle id must be in {1, 2}."
    assert r >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["logcosh", "exp", "cube"], "G function must be either \"logcosh\", \"exp\" or \"cube\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    # Patch Scikit-learn
    patch_sklearn()

    # Set random number generator
    prng = np.random.default_rng(seed)

    # 1. Load data
    # sub_emg = semg_bss.load_1dof(DATA_DIR, subject, session)
    # emg = sub_emg[finger][trial][128 * (muscle - 1):128 * muscle]  # load record for given finger, sample and muscle
    # emg_sep = np.zeros(shape=(n_comp, emg.shape[1]), dtype=float)

    t = np.linspace(0, 25, 25 * FS_EMG)

    # Matrix S with the original signals (n_components, n_samples)
    s = np.vstack([
        sine_wave(t, amp=1.5, freq=0.3, phase=np.pi),
        square_wave(t, amp=1, freq=0.5, phase=0),
        sawtooth_wave(t, amp=0.5, freq=0.7, phase=-np.pi)
    ])
    # s += 0.2 * np.random.normal(size=s.shape)  # noise
    std = s.std(axis=1)
    s1 = (s.T / std).T
    semg_bss.plot_signal(s1, FS_EMG)

    # Mixing matrix A
    a = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])  # Mixing matrix
    print(a)

    # Observations X (n_components, n_samples)
    emg = a @ s1
    n_samples = emg.shape[1]
    emg_sep = np.zeros(shape=(n_comp, n_samples), dtype=float)

    # Extend signal
    emg_ext = semg_bss.extend_signal(emg, r)

    # Prepare mini-batches with 75% overlap
    overlap = 3 * mb_size // 4
    sig_len = 1000 * n_samples // FS_EMG  # ms
    n_steps = ceil((sig_len - overlap) / (mb_size - overlap))
    mb_samples = FS_EMG * mb_size // 1000
    no_samples = FS_EMG * (mb_size - overlap) // 1000  # number of non-overlapping samples
    # Iterate over mini-batch steps
    for i in range(n_steps):
        print(f"{i + 1}/{n_steps}")
        start = i * no_samples
        end = n_samples if i == n_steps - 1 else start + mb_samples

        # 2. Garbage detection
        if not(semg_bss.garbage_detection(emg_ext[:, start:end], 2)):
            # 3. Whitening
            emg_white, white_mtx = semg_bss.whiten_signal(emg_ext[:, start:end])

            # 4. FastICA
            emg_sep_cur, sep_mtx = semg_bss.fast_ica(
                emg_white,
                n_comp,
                strategy,
                g_func,
                max_iter,
                ica_th,
                prng,
                verbose=True
            )

        # Save only non-overlapping data
        emg_sep[:, start + mb_samples - no_samples:end] = emg_sep_cur[:, mb_samples - no_samples:]
    semg_bss.plot_signal(emg_sep, FS_EMG)


if __name__ == "__main__":
    main()
