import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

import hyser

PATH_DATA = "/home/nihil/Scrivania/hyser_dataset"
FS_EMG = 2048


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=1, type=int, help="Subject id (in range [1, 20]")
    ap.add_argument("--session", default=1, type=int, help="Session id (in {1, 2})")
    ap.add_argument("--finger", default=2, type=int, help="Finger id (in {1, 2, 3, 4, 5})")
    ap.add_argument("--sample", default=1, type=int, help="Sample id (in {1, 2, 3})")
    ap.add_argument("--muscle", default=1, type=int, help="Muscle id (in {1, 2})")
    ap.add_argument("-r", default=4, type=int, help="Extension factor")
    ap.add_argument("--n_comp", default=300, type=int, help="Number of components to extract")
    ap.add_argument("--strategy", default="deflation", type=str, help="FastICA strategy")
    ap.add_argument("--g_func", default="logcosh", type=str, help="Non-quadratic function G")
    ap.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations")
    ap.add_argument("--ica_th", default=1e-4, type=float, help="Threshold for ICA")
    ap.add_argument("--sil_th", default=0.6, type=float, help="Threshold for SIL")
    args = vars(ap.parse_args())

    # Read input arguments
    subject = args["subject"]
    session = args["session"]
    finger = args["finger"]
    sample = args["sample"]
    muscle = args["muscle"]
    r = args["r"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    ica_th = args["ica_th"]
    sil_th = args["sil_th"]

    # Check input
    assert subject in range(1, 21), "Subject id must be in range [1, 20]."
    assert session in [1, 2], "Session id must be in {1, 2}."
    assert finger in [1, 2, 3, 4, 5], "Finger id must be in {1, 2, 3, 4, 5}."
    assert sample in [1, 2, 3], "Sample id must be in {1, 2, 3}."
    assert muscle in [1, 2], "Muscle id must be in {1, 2}."
    assert r >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["logcosh", "exp", "cube"], "G function must be either \"logcosh\", \"exp\" or \"cube\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    # 0. Load data
    sub_emg = hyser.load_1dof(PATH_DATA, subject, session)
    emg = sub_emg[finger][sample][128 * (muscle - 1):128 * muscle]  # load record for given finger, sample and muscle

    # 1. Preprocessing (extension + whitening)
    emg_white, white_mtx = hyser.whiten_signal(
        hyser.extend_signal(emg, r)
    )

    # 2. FastICA
    emg_sep, sep_mtx = hyser.fast_ica(emg_white, n_comp, strategy, g_func, max_iter, ica_th)

    # 3. Spike detection
    spike_train = hyser.spike_detection(emg_sep, sil_th)

    # 4. MU duplicates removal
    spike_train, emg_sep = hyser.replicas_removal(spike_train, emg_sep, FS_EMG)

    for i in range(0, 8):
        plt.subplot(8, 1, i + 1)
        plt.plot(spike_train[i])
    plt.show()


if __name__ == "__main__":
    main()
