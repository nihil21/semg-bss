import csv
import glob
import os
import re
import sys

import wfdb

PATH_SOURCE = "/data/physionet.org/files/hd-semg/1.0.0/pr_dataset"


def main():
    # Read input
    if len(sys.argv) < 3:
        sys.exit("Usage: python3 hyser_subset.py path_dest *[1, 34]")
    path_dest = sys.argv[1]
    gestures = [int(g) for g in sys.argv[2:]]
    for g in gestures:
        os.makedirs(os.path.join(path_dest, f"{g:02d}"), exist_ok=True)

    # Get list of folders
    folders = sorted(glob.glob(os.path.join(PATH_SOURCE, "*", "")))
    for folder in folders:
        # Get subject and session
        subject, session = re.findall("\d+", folder.split("/")[-2])

        for task_type in ("dynamic", "maintenance"):
            for sig_type in ["raw", "preprocess"]:
                # Get list of files and order them by sample number
                files = glob.glob(os.path.join(folder, f"{task_type}_{sig_type}_sample*.dat"))  # search only .dat files
                files = [
                    (int(re.findall("\d+", f.split("/")[-1])[0]), f)
                    for f in files
                ]
                files = sorted(files, key=lambda t: t[0])

                # Read labels
                label_file = os.path.join(folder, f"label_{task_type}.txt")
                with open(label_file, "r") as f:
                    labels = [int(c) for line in f for c in line.strip().split(",")]

                # Copy files corresponding to labels to new location
                for g in gestures:
                    gesture_folder = os.path.join(path_dest, f"{g:02d}")
                    k = 0
                    gesture_idx = [i + 1 for i, l in enumerate(labels) if l == g]
                    for i in gesture_idx:
                        for f_i, f in files:
                            if f_i == i:
                                # Create new file name
                                if task_type == "dynamic":  # 2 trials, 3 tasks
                                    trial = k // 3 + 1
                                    task = k % 3 + 1
                                else:  # 2 trials, 1 task
                                    trial = k % 2 + 1
                                    task = 1
                                k += 1
                                new_file_name = f"subject{subject}_session{session}_{task_type}_{sig_type}_trial{trial}_task{task}"
                        
                                # Read WFDB signal
                                signals, fields = wfdb.rdsamp(f[:-4])
                                # Remove unnecessary fields
                                fields.pop("sig_len")
                                fields.pop("n_sig")
                                # Write them to new directory
                                wfdb.wrsamp(
                                    new_file_name,
                                    p_signal=signals,
                                    write_dir=gesture_folder,
                                    **fields,
                                )
                        
                                # Write to CSV
                                # with open(os.path.join(gesture_folder, new_file_name), "w", newline="") as f:
                                #     writer = csv.writer(f)
                                #     writer.writerows(signals.T)


if __name__ == "__main__":
    main()
