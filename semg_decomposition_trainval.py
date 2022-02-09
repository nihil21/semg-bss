import argparse

import numpy as np
from sklearnex import patch_sklearn

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/hyser_dataset"
CACHE_DIR = "cache"

FS_EMG = 2048


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
    ap.add_argument("--train_size", default=100, type=int, help="Size of the training set (in ms).")

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
    train_size = args["train_size"]

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
    sub_emg = semg_bss.load_1dof(DATA_DIR, subject, session)
    emg = sub_emg[finger][trial][128 * (muscle - 1):128 * muscle]  # load record for given finger, sample and muscle
    n_samples = emg.shape[1]
    emg_sep = np.zeros(shape=(n_comp, n_samples), dtype=float)

    # Cut point between training and validation set
    cut = FS_EMG * train_size // 1000

    start = 0
    end = cut

    # 2. Preprocessing (extension + whitening)
    emg_ext = semg_bss.extend_signal(emg, r)
    emg_white, white_mtx = semg_bss.whiten_signal(emg_ext[:, start:end])

    # 3. FastICA
    cur_emg_sep, sep_mtx = semg_bss.fast_ica(emg_white, n_comp, strategy, g_func, max_iter, ica_th, prng, verbose=True)
    emg_sep[:, start:end] = cur_emg_sep

    start = end
    end = n_samples

    # 2b. Preprocessing (extension + whitening) with precomputed whitening matrix
    emg_mean = np.mean(emg_ext[:, start:end], axis=1, keepdims=True)
    emg_center = emg_ext[:, start:end] - emg_mean
    emg_white = white_mtx @ emg_center

    # 3b. FastICA with precomputed weight matrix
    cur_emg_sep = sep_mtx.T @ emg_white
    emg_sep[:, start:end] = cur_emg_sep

    # 4. Spike detection
    spike_train = semg_bss.spike_detection(emg_sep, sil_th, seed=seed, verbose=True)

    # 5. MU duplicates removal
    valid_index = semg_bss.replicas_removal(spike_train, emg_sep, FS_EMG)
    spike_train = spike_train[valid_index]
    emg_sep = emg_sep[valid_index]
    print("N. components extracted:", spike_train.shape[0])

    # 6. Compute silhouette
    sil = semg_bss.silhouette(emg_sep, FS_EMG, seed=seed, verbose=True)
    n_mu = sil[sil > sil_th].shape[0]
    avg_sil = sil[sil > sil_th].mean()

    print(f"{n_mu} MUs extracted with an average silhouette score of {avg_sil:.4f}")

    emg_sep_valid = emg_sep[sil > sil_th]
    spike_train_valid = spike_train[sil > sil_th]
    semg_bss.plot_signals(emg_sep_valid[:25], spike_train_valid[:25], FS_EMG, n_cols=3, fig_size=(50, 70))


if __name__ == "__main__":
    main()
