from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt

import semg_bss

DATA_DIR = "/home/nihil/Scrivania/simulated_dataset"
MAX_COMP = 25
SIL = 0.85


def compute_stats(
        firings: np.ndarray,
        gt_firings: np.ndarray,
        win_len: float = 0.01
) -> tuple[bool, float | None, float | None, float | None]:
    sync: list[float] = []
    for f in firings:
        fire_interval = gt_firings - f
        idx = np.flatnonzero((fire_interval >= -win_len) & (fire_interval <= win_len))
        if idx.size != 0:
            sync.extend(fire_interval[idx])

    # Compute histogram of relative timing
    hist, bin_edges = np.histogram(sync, bins="auto")
    max_bin = hist.argmax()
    common_idx = np.flatnonzero((sync >= bin_edges[max_bin]) & (sync <= bin_edges[max_bin + 1]))

    # If less than 30% of firings are synchronized, discard MU
    min_perc = 0.3
    if common_idx.size / firings.size < min_perc:
        return False, None, None, None

    tp = common_idx.size  # true positive
    fp = firings.size - tp  # false positive
    fn = gt_firings.size - tp  # false negative

    return True, tp, fp, fn


def main():
    logging.basicConfig(encoding='utf-8', level=logging.WARNING, filemode="w")

    for mvc in (10, 30, 50):
        print(f"----- MVC = {mvc}% -----")

        avg_roa = 0
        avg_precision = 0
        avg_recall = 0
        avg_valid_n_mu = 0
        avg_tot_n_mu = 0
        s = 0
        for emg, gt_firings, fs_emg in semg_bss.simulated.load_simulated(DATA_DIR, mvc=mvc):
            # semg_bss.plot_signal(emg[:15], fs=fs_emg, n_cols=5, fig_size=(20, 8))
            # semg_bss.raster_plot(
            #     gt_firings,
            #     title=f"Simulated spikes (sample {s + 1}, MVC {MVC}%)",
            #     sig_len=16,
            #     fig_size=(20, 12)
            # )

            emg_separator = semg_bss.EMGSeparator(
                max_comp=MAX_COMP,
                fs=fs_emg,
                f_e=16,
                sil_th=SIL,
                seed=42
            )

            firings = emg_separator.calibrate(emg)
            # semg_bss.raster_plot(
            #     firings,
            #     title=f"Estimated spikes (sample {s + 1}, MVC {MVC}%)",
            #     sig_len=16,
            #     fig_size=(20, 12)
            # )

            # Convert to dict MU -> spikes
            firings_dict = {k: v.to_numpy() for k, v in firings.groupby("MU index")["Firing time"]}
            gt_firings_dict = {k: v.to_numpy() for k, v in gt_firings.groupby("MU index")["Firing time"]}

            # A spike train is valid only if it shares at least 30% of firings with ground truth
            valid_firings_dict = {}
            for i in firings_dict.keys():
                for j in gt_firings_dict.keys():
                    valid, tp, fp, fn = compute_stats(firings_dict[i], gt_firings_dict[j], win_len=0.005)
                    if valid:
                        if i not in valid_firings_dict.keys():
                            valid_firings_dict[i] = {}
                        roa = tp / (tp + fn + fp)
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        valid_firings_dict[i][j] = (roa, precision, recall)

            cur_valid_n_mu = len(valid_firings_dict.keys())
            cur_tot_n_mu = len(firings_dict.keys())
            print(f"Extracted {cur_valid_n_mu} valid MUs out of {cur_tot_n_mu} detected for sample {s + 1}.")

            cur_avg_roa = 0
            cur_avg_precision = 0
            cur_avg_recall = 0
            max_mus = []
            for mu, measures in valid_firings_dict.items():
                max_mu = max(measures, key=measures.get)
                if max_mu in max_mus:  # ground truth MU already identified
                    continue
                max_mus.append(max_mu)
                stats = measures[max_mu]
                print(f"Estimated MU {mu} <-> ground truth MU {max_mu}")
                print(f"\tRoA: {stats[0]:.2%}\tPrecision: {stats[1]:.2%}\t Recall: {stats[2]:.2%}")
                cur_avg_roa += stats[0]
                cur_avg_precision += stats[1]
                cur_avg_recall += stats[2]

            cur_avg_roa /= cur_valid_n_mu
            cur_avg_precision /= cur_valid_n_mu
            cur_avg_recall /= cur_valid_n_mu
            print(f"Average RoA for sample {s + 1}: {cur_avg_roa:.2%}")
            print(f"Average precision for sample {s + 1}: {cur_avg_precision:.2%}")
            print(f"Average recall for sample {s + 1}: {cur_avg_recall:.2%}")

            plt.figure(figsize=(20, 12))
            plt.title(f"Estimated spikes (sample {s + 1}, MVC {mvc}%, SIL {SIL})")
            for i, (mu, measures) in enumerate(valid_firings_dict.items()):
                max_mu = max(measures, key=measures.get)
                stats = measures[max_mu]
                plt.scatter(x=firings_dict[mu], y=[i + 0.9] * len(firings_dict[mu]), marker="|", color="k")
                plt.scatter(x=gt_firings_dict[max_mu], y=[i + 1.1] * len(gt_firings_dict[max_mu]), marker="|", color="r")
                plt.text(x=17, y=i + 0.8, s=f"RoA = {stats[0]:.2%}\nPrecision = {stats[1]:.2%}\nRecall = {stats[2]:.2%}")
            plt.show()

            avg_roa += cur_avg_roa
            avg_precision += cur_avg_precision
            avg_recall += cur_avg_recall
            avg_valid_n_mu += cur_valid_n_mu
            avg_tot_n_mu += cur_tot_n_mu
            s += 1

        avg_roa /= s
        avg_precision /= s
        avg_recall /= s
        avg_valid_n_mu /= s
        avg_tot_n_mu /= s
        print(f"Extracted {avg_valid_n_mu} valid MUs out of {avg_tot_n_mu} detected on average.")
        print(f"Average RoA: {avg_roa:.2%}")
        print(f"Average precision: {avg_precision:.2%}")
        print(f"Average recall: {avg_recall:.2%}")


if __name__ == "__main__":
    main()
