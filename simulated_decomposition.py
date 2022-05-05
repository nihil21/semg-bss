from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/simulated_dataset"


def sync_correlation(
    firings_ref: np.ndarray,
    firings_sec: np.ndarray,
    time_lock: float,
    min_perc: float,
    win_len: float = 0.01
) -> tuple[bool, int | None]:
    sync: list[float] = []
    for firing_ref in firings_ref:
        fire_interval = firings_sec - firing_ref
        idx = np.flatnonzero(
            (fire_interval >= -win_len) & (fire_interval <= win_len)
        )
        if idx.size != 0:
            sync.extend(fire_interval[idx])

    # Compute histogram of relative timing
    hist, bin_edges = np.histogram(sync)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    max_bin = hist.argmax()
    common_idx = np.flatnonzero(
        (sync >= bin_centers[max_bin] - time_lock) & (sync <= bin_centers[max_bin] + time_lock)
    )

    in_sync = common_idx.size / firings_ref.size > min_perc
    n_common = common_idx.size if in_sync else None

    return in_sync, n_common


def compute_stats(
    gt_firings: np.ndarray,
    firings: np.ndarray,
    time_lock: float,
    min_perc=0.3,
    win_len: float = 0.01
) -> tuple[bool, float | None, float | None, float | None]:
    in_sync1, n_common = sync_correlation(gt_firings, firings, time_lock, min_perc, win_len)
    in_sync2, _ = sync_correlation(firings, gt_firings, time_lock, min_perc, win_len)

    if not (in_sync1 and in_sync2):
        return False, None, None, None

    tp = n_common  # true positive
    fp = firings.size - tp  # false positive
    fn = gt_firings.size - tp  # false negative

    return True, tp, fp, fn


def main():
    logging.basicConfig(encoding='utf-8', level=logging.INFO, filename="../semg_bss.log", filemode="w")

    max_comp_dict = {
        10: 300,
        30: 400,
        50: 500
    }

    for mvc in (10, 30, 50):
        # for snr in (10, 20, 30):
        # print(f"----- MVC = {mvc}%, SNR = {snr} -----")
        print(f"----- MVC = {mvc}% -----")

        roa_avg_list = []
        roa_std_list = []
        precision_avg_list = []
        precision_std_list = []
        recall_avg_list = []
        recall_std_list = []
        valid_n_mu_list = []
        tot_n_mu_list = []
        s = 0
        for emg, gt_firings, fs_emg in semg_bss.simulated.load_simulated(DATA_DIR, mvc=mvc):  # , snr=snr):
            # Filter signal with 20-500 Hz band-pass filter
            emg = semg_bss.preprocessing.filter_signal(
                emg,
                fs=fs_emg,
                min_freq=20,
                max_freq=500,
                order=1
            )

            # Create separator and calibrate it
            emg_separator = semg_bss.EMGSeparator(
                max_sources=max_comp_dict[mvc],
                samp_freq=fs_emg,
                f_e=16,
                seed=42
            )
            firings = emg_separator.calibrate(
                emg,
                min_spike_pps=8,
                # max_spike_pps=16
            )

            # Convert to dict MU -> spikes
            firings_dict = {k: v.to_numpy() for k, v in firings.groupby("MU index")["Firing time"]}
            gt_firings_dict = {k: v.to_numpy() for k, v in gt_firings.groupby("MU index")["Firing time"]}

            # A spike train is valid only if it shares at least 30% of firings with ground truth
            valid_firings_dict: dict[int, tuple[int, float, float, float]] = {}
            identified_mus_dict: dict[int, tuple[int, float]] = {}
            for i in firings_dict.keys():
                for j in gt_firings_dict.keys():
                    valid, tp, fp, fn = compute_stats(gt_firings_dict[j], firings_dict[i], time_lock=5e-4)
                    if valid:
                        # Compute metrics
                        roa = tp / (tp + fn + fp)
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)

                        # If the ground truth MU was already identified, consider the association
                        # with the estimated MU having the highest RoA
                        if j in identified_mus_dict.keys():
                            if roa > identified_mus_dict[j][1]:
                                valid_firings_dict[i] = (j, roa, precision, recall)
                                identified_mus_dict[j] = (i, roa)
                        else:
                            valid_firings_dict[i] = (j, roa, precision, recall)
                            identified_mus_dict[j] = (i, roa)

            cur_valid_n_mu = len(valid_firings_dict.keys())
            cur_tot_n_mu = len(firings_dict.keys())
            print(f"Extracted {cur_valid_n_mu} valid MUs out of {cur_tot_n_mu} detected for sample {s + 1}.")

            cur_roa_list = []
            cur_precision_list = []
            cur_recall_list = []
            for mu, (gt_mu, roa, precision, recall) in valid_firings_dict.items():
                print(f"Estimated MU {mu} <-> ground truth MU {gt_mu}")
                print(f"\tRoA: {roa:.2%}\tPrecision: {precision:.2%}\t Recall: {recall:.2%}")
                cur_roa_list.append(roa)
                cur_precision_list.append(precision)
                cur_recall_list.append(recall)

            cur_roa_avg = np.mean(cur_roa_list)
            cur_roa_std = np.std(cur_roa_list)
            cur_precision_avg = np.mean(cur_precision_list)
            cur_precision_std = np.std(cur_precision_list)
            cur_recall_avg = np.mean(cur_recall_list)
            cur_recall_std = np.std(cur_recall_list)
            print(f"Average RoA for sample {s + 1}: {cur_roa_avg:.2f} +- {cur_roa_std:.2f}")
            print(f"Average precision for sample {s + 1}: {cur_precision_avg:.2f} +- {cur_precision_std:.2f}")
            print(f"Average recall for sample {s + 1}: {cur_recall_avg:.2f} +- {cur_recall_std:.2f}")

            plt.figure(figsize=(20, 20))
            plt.title(f"Estimated spikes (sample {s + 1}, MVC {mvc}%)")
            for i, (mu, (gt_mu, roa, precision, recall)) in enumerate(valid_firings_dict.items()):
                plt.scatter(
                    x=firings_dict[mu],
                    y=[i - 0.1] * len(firings_dict[mu]),
                    marker="|",
                    color="k"
                )
                plt.scatter(
                    x=gt_firings_dict[gt_mu],
                    y=[i + 0.1] * len(gt_firings_dict[gt_mu]),
                    marker="|",
                    color="r"
                )
                plt.text(
                    x=17,
                    y=i,
                    s=f"RoA = {roa:.2%}\nPrecision = {precision:.2%}\nRecall = {recall:.2%}"
                )
            plt.yticks(range(cur_valid_n_mu))
            plt.show()

            roa_avg_list.append(cur_roa_avg)
            roa_std_list.append(cur_roa_std)
            precision_avg_list.append(cur_precision_avg)
            precision_std_list.append(cur_precision_std)
            recall_avg_list.append(cur_recall_avg)
            recall_std_list.append(cur_recall_std)
            valid_n_mu_list.append(cur_valid_n_mu)
            tot_n_mu_list.append(cur_tot_n_mu)
            s += 1

        roa_avg = np.mean(roa_avg_list)
        roa_std = np.sqrt(
            np.mean(
                list(map(lambda x: x**2, roa_std_list))
            )
        )
        precision_avg = np.mean(precision_avg_list)
        precision_std = np.sqrt(
            np.mean(
                list(map(lambda x: x ** 2, precision_std_list))
            )
        )
        recall_avg = np.mean(recall_avg_list)
        recall_std = np.sqrt(
            np.mean(
                list(map(lambda x: x ** 2, recall_std_list))
            )
        )
        valid_n_mu_avg = np.mean(valid_n_mu_list)
        valid_n_mu_std = np.std(valid_n_mu_list)
        tot_n_mu_avg = np.mean(tot_n_mu_list)
        tot_n_mu_std = np.std(tot_n_mu_list)
        print(f"Extracted {valid_n_mu_avg:.2f} +- {valid_n_mu_std:.2f} valid MUs "
              f"out of {tot_n_mu_avg:.2f} +- {tot_n_mu_std:.2f} detected on average.")
        print(f"Average RoA: {roa_avg:.2f} +- {roa_std:.2f}")
        print(f"Average precision: {precision_avg:.2f} +- {precision_std:.2f}")
        print(f"Average recall: {recall_avg:.2f} +- {recall_std:.2f}")


if __name__ == "__main__":
    main()
