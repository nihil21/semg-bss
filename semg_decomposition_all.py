import argparse
import time

import numpy as np
from sklearnex import patch_sklearn

import semg_bss

PATH_DATA = "/data/physionet.org/files/hd-semg/1.0.0/"
FS_EMG = 2048


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", default=4, type=int, help="Extension factor")
    ap.add_argument("--n_comp", default=300, type=int, help="Number of components to extract")
    ap.add_argument("--strategy", default="deflation", type=str, help="FastICA strategy")
    ap.add_argument("--g_func", default="logcosh", type=str, help="Non-quadratic function G")
    ap.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations")
    ap.add_argument("--ica_th", default=1e-4, type=float, help="Threshold for ICA")
    ap.add_argument("--sil_th", default=0.6, type=float, help="Threshold for SIL")
    ap.add_argument("--seed", default=None, type=int, help="Seed for PRNG.")
    args = vars(ap.parse_args())

    # Read input arguments
    r = args["r"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    ica_th = args["ica_th"]
    sil_th = args["sil_th"]
    seed = args["seed"]

    # Check input
    assert r >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["logcosh", "exp", "cube"], "G function must be either \"logcosh\", \"exp\" or \"cube\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    # Patch Scikit-learn
    patch_sklearn()

    # Set random number generator
    prng = np.random.default_rng(seed)

    # Iterate over subjects
    for subject in range(1, 21):
        # Iterate over sessions
        for session in [1, 2]:
            # 0. Load data
            print("Loading data...\t\t", end="", flush=True)
            start = time.time()
            sub_emg = semg_bss.load_1dof(PATH_DATA, subject, session)
            stop = time.time()
            print(f"Done [elapsed: {stop - start:.2f} s]")
            for finger in [1, 2, 3, 4, 5]:
                for sample in [1, 2, 3]:
                    for muscle in [1, 2]:
                        # Load record for given finger, sample and muscle
                        emg = sub_emg[finger][sample][128 * (muscle - 1):128 * muscle]
                        # 1. Preprocessing (extension + whitening)
                        emg_white, white_mtx = semg_bss.whiten_signal(
                            semg_bss.extend_signal(emg, r)
                        )
                        # 2. FastICA
                        emg_sep, sep_mtx = semg_bss.fast_ica(
                            emg_white,
                            n_comp,
                            strategy,
                            g_func,
                            max_iter,
                            ica_th,
                            prng,
                            verbose=True
                        )
                        # 4. Spike detection
                        spike_train = semg_bss.spike_detection(
                            emg_sep,
                            sil_th,
                            seed=seed,
                            verbose=True
                        )
                        # 5. MU duplicates removal
                        valid_index = semg_bss.replicas_removal(spike_train, emg_sep, FS_EMG)
                        spike_train = spike_train[valid_index]
                        emg_sep = emg_sep[valid_index]
                        # 6. Compute silhouette
                        sil = semg_bss.silhouette(emg_sep, FS_EMG, seed=seed, verbose=True)
                        n_mu = sil[sil > sil_th].shape[0]
                        avg_sil = sil[sil > sil_th].mean()
                        emg_sep_valid = emg_sep[sil > sil_th]
                        spike_train_valid = spike_train[sil > sil_th]


if __name__ == "__main__":
    main()
