import argparse
import time

import hyser
import numpy as np


PATH_DATA = "/data/physionet.org/files/hd-semg/1.0.0/"
SAMP_FREQ = 2048


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
    args = vars(ap.parse_args())

    # Read input arguments
    r = args["r"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    ica_th = args["ica_th"]
    sil_th = args["sil_th"]

    # Check input
    assert r >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["logcosh", "exp", "cube"], "G function must be either \"logcosh\", \"exp\" or \"cube\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    # Iterate over subjects
    for subject in range(1, 21):
        # Iterate over sessions
        for session in [1, 2]:
            # 0. Load data
            print("Loading data...\t\t", end="", flush=True)
            start = time.time()
            sub_emg = hyser.load_1dof(PATH_DATA, subject, session)
            stop = time.time()
            print(f"Done [elapsed: {stop - start:.2f} s]")
            for finger in [1, 2, 3, 4, 5]:
                for sample in [1, 2, 3]:
                    for muscle in [1, 2]:
                        # Load record for given finger, sample and muscle
                        emg = sub_emg[finger][sample][128 * (muscle - 1):128 * muscle]
                        # 1. Preprocessing (extension + whitening)
                        start = time.time()
                        print("Preprocessing...\t", end="", flush=True)
                        emg_white, white_mtx = hyser.preprocessing(emg, r)
                        stop = time.time()
                        print(f"Done [elapsed: {stop - start:.2f} s]")
                        # 2. FastICA
                        start = time.time()
                        print("FastICA...", end="\t", flush=True)
                        emg_sep, sep_mtx = hyser.fast_ica(emg_white, n_comp, strategy, g_func, max_iter, ica_th)
                        stop = time.time()
                        print(f"Done [elapsed: {stop - start:.2f} s]")


if __name__ == "__main__":
    main()
