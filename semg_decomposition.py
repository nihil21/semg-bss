import argparse
import logging

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/hyser_dataset"

FS_EMG = 2048


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=0, type=int, help="Subject id (in range [0, 19]")
    ap.add_argument("--session", default=0, type=int, help="Session id (in {0, 1})")
    ap.add_argument("--combination", default=0, type=int, help="Combination id (in range [0, 14])")
    ap.add_argument("--trial", default=0, type=int, help="Trial id (in {0, 1})")
    ap.add_argument("--f_e", default=4, type=int, help="Extension factor")
    ap.add_argument("--n_comp", default=100, type=int, help="Number of components to extract")
    ap.add_argument("--strategy", default="deflation", type=str, help="FastICA strategy")
    ap.add_argument("--g_func", default="tanh", type=str, help="Contrast function G")
    ap.add_argument("--max_iter", default=100, type=int, help="Maximum number of iterations")
    ap.add_argument("--conv_th", default=1e-4, type=float, help="Threshold for convergence")
    ap.add_argument("--sil_th", default=0.9, type=float, help="Threshold for SIL")
    ap.add_argument("--seed", default=None, type=int, help="Seed for PRNG")
    args = vars(ap.parse_args())

    # Read input arguments
    subject = args["subject"]
    session = args["session"]
    combination = args["combination"]
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
    assert subject in range(20), "Subject id must be in range [0, 19]."
    assert session in [0, 1], "Session id must be in {0, 1}."
    assert combination in range(15), "Combination id must be in range [0, 14]."
    assert trial in [0, 1], "Sample id must be in {0, 1}."
    assert f_e >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["tanh", "exp", "cube"], "G function must be either \"tanh\", \"exp\" or \"cube\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    # 1. Load sEMG data
    sub_emg = semg_bss.hyser.load_ndof(DATA_DIR, subject, session, sig_type="preprocess", verbose=True)
    emg = sub_emg[combination, trial]
    semg_bss.plot_signal(emg[:15], fig_size=(800, 1200), fs=FS_EMG, n_cols=5)

    logging.basicConfig(encoding='utf-8', level=logging.INFO, filemode="w")

    emg_separator = semg_bss.EmgSeparator(n_comp, FS_EMG, f_e, g_func, conv_th, sil_th, max_iter, seed=seed)
    firings = emg_separator.fit_transform(emg)
    semg_bss.raster_plot(firings, fig_size=(800, 1200))


if __name__ == "__main__":
    main()
