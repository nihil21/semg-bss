import argparse

import semg_bss

DATA_DIR = "/data/physionet.org/files/hd-semg/1.0.0"
OUT_DIR = "NDoF_results"

FS_EMG = 2048


def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("--subset", required=True, type=str, help="Name of the Hyser's subset to analyze")
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
    # subset = args["subset"]
    f_e = args["f_e"]
    n_comp = args["n_comp"]
    strategy = args["strategy"]
    g_func = args["g_func"]
    max_iter = args["max_iter"]
    conv_th = args["conv_th"]
    sil_th = args["sil_th"]
    seed = args["seed"]

    # Check input
    # assert subset in ["pr", "mvc", "1dof", "ndof", "random"], \
    #     "Subset must be either \"pr\", \"mvc\", \"1dof\", \"ndof\" or \"random\"."
    assert f_e >= 0, "Extension factor must be non-negative."
    assert n_comp > 0, "Number of components must be positive."
    assert strategy in ["deflation", "parallel"], "FastICA strategy must be either \"deflation\" or \"parallel\"."
    assert g_func in ["logcosh", "exp", "skew"], "G function must be either \"logcosh\", \"exp\" or \"skew\"."
    assert max_iter > 0, "Number of maximum iterations must be positive."

    emg_separator = semg_bss.EmgSeparator(n_comp, FS_EMG, f_e, g_func, conv_th, sil_th, max_iter, seed=seed)

    for subject in range(20):
        for session in range(2):
            sub_emg = semg_bss.hyser.load_ndof(DATA_DIR, subject, session, sig_type="preprocess", verbose=True)
            for combination in range(15):
                for trial in range(2):
                    emg = sub_emg[combination, trial]
                    firings = emg_separator.fit_transform(emg)
                    firings.to_parquet(
                        path=f"firings_{subject + 1:02d}_{session + 1}_{combination + 1}_{trial + 1}.gzip",
                        compression="gzip"
                    )


if __name__ == "__main__":
    main()
