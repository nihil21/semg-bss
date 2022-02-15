import argparse

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LinearRegression
from sklearnex import patch_sklearn

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/hyser_dataset"

FS_FORCE = 100
FS_EMG = 2048


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=0, type=int, help="Subject id (in range [0, 19]")
    ap.add_argument("--session", default=0, type=int, help="Session id (in {0, 1})")
    ap.add_argument("--finger", default=0, type=int, help="Finger id (in {0, 1, 2, 3, 4})")
    ap.add_argument("--trial", default=0, type=int, help="Trial id (in {0, 1, 2})")
    ap.add_argument("-r", default=4, type=int, help="Extension factor")
    ap.add_argument("--n_comp", default=300, type=int, help="Number of components to extract")
    ap.add_argument("--strategy", default="deflation", type=str, help="FastICA strategy")
    ap.add_argument("--g_func", default="logcosh", type=str, help="Non-quadratic function G")
    ap.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations")
    ap.add_argument("--ica_th", default=1e-4, type=float, help="Threshold for ICA")
    ap.add_argument("--sil_th", default=0.6, type=float, help="Threshold for SIL")
    ap.add_argument("--win_size", default=0.02, type=float, help="Window size for feature extraction")
    ap.add_argument("--step_len", default=0.02, type=float, help="Step length for feature extraction")
    ap.add_argument("--seed", default=None, type=int, help="Seed for PRNG")
    args = vars(ap.parse_args())

    # Read input arguments
    subject = args["subject"]
    session = args["session"]
    finger = args["finger"]
    trial = args["trial"]
    r = args["r"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    ica_th = args["ica_th"]
    sil_th = args["sil_th"]
    win_size = args["win_size"]
    step_len = args["step_len"]
    seed = args["seed"]

    # Check input
    assert subject in range(20), "Subject id must be in range [0, 19]."
    assert session in [0, 1], "Session id must be in {0, 1}."
    assert finger in [0, 1, 2, 3, 4], "Finger id must be in {0, 1, 2, 3, 4}."
    assert trial in [0, 1, 2], "Sample id must be in {0, 1, 2}."
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
    sub_emg = semg_bss.hyser.load_1dof(DATA_DIR, subject, session, sig_type="raw")
    sub_emg = sub_emg[finger, trial]
    # 1b. Load force data
    sub_force = semg_bss.hyser.load_1dof(DATA_DIR, subject, session, sig_type="force")
    sub_mvc = semg_bss.hyser.get_mvc(DATA_DIR, subject, session)
    force_norm = semg_bss.hyser.normalize_force(sub_force, sub_mvc)
    force_preprocessed = semg_bss.hyser.preprocess_force(force_norm, win_size, step_len, FS_FORCE, FS_EMG)
    force_preprocessed = force_preprocessed[finger, trial]

    spike_train_valid = []
    sil_valid = []
    n_mu = {}
    for muscle_idx in (1, 2):
        emg = sub_emg[128 * (muscle_idx - 1):128 * muscle_idx]

        # 2. Preprocessing (extension + whitening)
        emg_ext = semg_bss.preprocessing.extend_signal(emg, r)
        emg_center, _ = semg_bss.preprocessing.center_signal(emg_ext)
        emg_white, _ = semg_bss.preprocessing.whiten_signal(emg_center)

        # 3. FastICA
        emg_sep, sep_mtx = semg_bss.fast_ica(emg_white, n_comp, strategy, g_func, max_iter, ica_th, prng, verbose=True)

        # 4. Spike detection
        spike_train = semg_bss.postprocessing.spike_detection(emg_sep, sil_th, seed=seed, verbose=True)

        # 5. MU duplicates removal
        valid_idx = semg_bss.postprocessing.replicas_removal(spike_train, emg_sep, FS_EMG)
        spike_train = spike_train[valid_idx]
        emg_sep = emg_sep[valid_idx]

        # 6. Compute silhouette
        sil = semg_bss.metrics.silhouette(emg_sep, FS_EMG, seed=seed, verbose=True)
        valid_idx = np.nonzero(sil > sil_th)[0]
        spike_train_valid.append(spike_train[valid_idx])
        sil_valid.append(sil[valid_idx])
        n_mu[muscle_idx] = spike_train[valid_idx].shape[0]

    spike_train_valid = np.concatenate(spike_train_valid, axis=0)
    sil_valid = np.concatenate(sil_valid, axis=0)
    n_mu_tot = spike_train_valid.shape[0]
    correlation = None
    if n_mu_tot != 0:
        firing_rate, time_win = semg_bss.hyser.estimate_firing_rate(
            spike_train_valid,
            np.arange(n_mu_tot),
            win_size,
            step_len,
            FS_EMG
        )
        # Filter force signal
        wn = 10 / (1 / (time_win[1] - time_win[0]) / 2)
        sos = butter(8, wn, "low", output="sos")
        firing_rate = sosfiltfilt(sos, firing_rate)
        # Remove first and last 1s of signals, and perform linear regression
        mlr = LinearRegression()
        mlr.fit(firing_rate[:, 50:-51].T, force_preprocessed[finger, 50:-51].T)
        parameter = mlr.coef_
        force_regress = parameter.reshape(1, -1) @ firing_rate
        corr_mat = np.corrcoef(np.stack([force_regress[0, 50:-51], force_preprocessed[finger, 50:-51]]))
        correlation = corr_mat[0, 1]

    print(f"N. MU extracted: {n_mu[1]} and {n_mu[2]} for extensor and flexor muscles, respectively.")
    print("Average silhouette:", sil_valid.mean())
    print("Correlation with GT force:", correlation)


if __name__ == "__main__":
    main()
