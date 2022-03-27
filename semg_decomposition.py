import argparse
import logging

import numpy as np

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/hyser_dataset"

FS_EMG = 2048


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=1, type=int, help="Subject id (in range [1, 20]")
    ap.add_argument("--session", default=1, type=int, help="Session id (in {1, 2})")
    ap.add_argument("--task", default=1, type=int, help="Trial id (in {1, 3})")
    ap.add_argument("--trial", default=1, type=int, help="Trial id (in {1, 2})")
    ap.add_argument("--f_e", default=4, type=int, help="Extension factor")
    ap.add_argument("--n_comp", default=100, type=int, help="Number of components to extract")
    ap.add_argument("--strategy", default="deflation", type=str, help="FastICA strategy")
    ap.add_argument("--g_func", default="logcosh", type=str, help="Contrast function G")
    ap.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations")
    ap.add_argument("--conv_th", default=1e-4, type=float, help="Threshold for convergence")
    ap.add_argument("--sil_th", default=0.9, type=float, help="Threshold for SIL")
    ap.add_argument("--seed", default=None, type=int, help="Seed for PRNG")
    args = vars(ap.parse_args())

    # Read input arguments
    subject = args["subject"]
    session = args["session"]
    task = args["task"]
    trial = args["trial"]
    f_e = args["f_e"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    conv_th = args["conv_th"]
    sil_th = args["sil_th"]
    seed = args["seed"]

    # Check input
    assert subject in range(1, 21), "Subject id must be in range [1, 20]."
    assert session in (1, 2), "Session id must be in {1, 2}."
    assert task in (1, 2, 3), "Task id must be in {1, 2, 3}."
    assert trial in (1, 2), "Trial id must be in {1, 2}."
    assert f_e >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["logcosh", "exp", "skew"], "G function must be either \"logcosh\", \"exp\" or \"skew\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    # 1. Load sEMG data
    emg1 = semg_bss.hyser.load_pr(
        DATA_DIR,
        gesture=1,
        subject=subject,
        session=session,
        task=task,
        trial=1,
        task_type="maintenance",
        sig_type="preprocess"
    )
    semg_bss.plot_signal(emg1[:15], fs=FS_EMG, n_cols=5, fig_size=(20, 8))
    emg2 = semg_bss.hyser.load_pr(
        DATA_DIR,
        gesture=2,
        subject=subject,
        session=session,
        task=task,
        trial=1,
        task_type="maintenance",
        sig_type="preprocess"
    )
    semg_bss.plot_signal(emg2[:15], fs=FS_EMG, n_cols=5, fig_size=(20, 8))

    emg_train = np.stack([emg1, emg2], axis=0)
    # emg_train = np.concatenate([emg1, emg2], axis=-1)

    logging.basicConfig(encoding='utf-8', level=logging.INFO, filemode="w")

    emg_separator = semg_bss.EmgSeparator(
        n_comp,
        FS_EMG,
        f_e,
        g_func,
        conv_th,
        sil_th,
        max_iter,
        seed=seed
    )
    emg_separator = emg_separator.fit(emg_train)
    firings = emg_separator.transform(emg_train)
    semg_bss.raster_plot(
        list(firings),
        title=["Neural activity for gesture 1", "Neural activity for gesture 2"],
        sig_len=emg1.shape[1] / FS_EMG,
        n_cols=2,
        fig_size=(20, 12)
    )


if __name__ == "__main__":
    main()
