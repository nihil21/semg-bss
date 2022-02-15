import argparse

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import linregress
from sklearnex import patch_sklearn

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/simulated_dataset"


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=0, type=int, help="Subject id (in range [0, 14]")
    ap.add_argument("--mvc", default=10, type=int, help="MVC value (in {10, 30, 50 n})")
    ap.add_argument("-r", default=4, type=int, help="Extension factor")
    ap.add_argument("--n_comp", default=300, type=int, help="Number of components to extract")
    ap.add_argument("--strategy", default="deflation", type=str, help="FastICA strategy")
    ap.add_argument("--g_func", default="logcosh", type=str, help="Non-quadratic function G")
    ap.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations")
    ap.add_argument("--ica_th", default=1e-4, type=float, help="Threshold for ICA")
    ap.add_argument("--sil_th", default=0.6, type=float, help="Threshold for SIL")
    ap.add_argument("--seed", default=None, type=int, help="Seed for PRNG")
    args = vars(ap.parse_args())

    # Read input arguments
    subject = args["subject"]
    mvc = args["mvc"]
    r = args["r"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    ica_th = args["ica_th"]
    sil_th = args["sil_th"]
    seed = args["seed"]

    # Check input
    assert subject in range(15), "Subject id must be in range [0, 14]."
    assert mvc in [10, 30, 50], "MVC value must be in {10, 30, 50}."
    assert r >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["logcosh", "exp", "cube"], "G function must be either \"logcosh\", \"exp\" or \"cube\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    # Patch Scikit-learn
    patch_sklearn()

    # Set random number generator
    prng = np.random.default_rng(seed)

    # 1a. Load sEMG data
    emg, gt_spikes, fs_emg = semg_bss.simulated.load_semg(DATA_DIR, subject, mvc)
    semg_bss.plot_signal(emg[:50], fs_emg, n_cols=2, fig_size=(50, 70))
    semg_bss.plot_signal(gt_spikes[:50], fs_emg, n_cols=2, fig_size=(50, 70))

    # 2. Preprocessing (extension + whitening)
    emg_ext = semg_bss.preprocessing.extend_signal(emg, r)
    emg_center, _ = semg_bss.preprocessing.center_signal(emg_ext)
    emg_white, _ = semg_bss.preprocessing.whiten_signal(emg_center)

    # 3. FastICA
    emg_sep, sep_mtx = semg_bss.fast_ica(emg_white, n_comp, strategy, g_func, max_iter, ica_th, prng, verbose=True)

    # 4. Spike detection
    spike_train = semg_bss.postprocessing.spike_detection(emg_sep, sil_th, seed=seed, verbose=True)

    # 5. MU duplicates removal
    valid_idx = semg_bss.postprocessing.replicas_removal(spike_train, emg_sep, fs_emg)
    spike_train = spike_train[valid_idx]
    emg_sep = emg_sep[valid_idx]
    n_mu = spike_train.shape[0]

    print(f"{n_mu} MUs extracted after replicas removal.")

    # 6. Compute silhouette
    sil = semg_bss.metrics.silhouette(emg_sep, fs_emg, seed=seed, verbose=True)
    n_mu = sil[sil > sil_th].shape[0]
    avg_sil = sil[sil > sil_th].mean()

    print(f"{n_mu} MUs extracted with an average silhouette score of {avg_sil:.4f}")

    semg_bss.plot_signals(emg_sep[:25], spike_train[:25], fs_emg, n_cols=3, fig_size=(50, 70))


if __name__ == "__main__":
    main()
