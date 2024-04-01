"""
Data Preparation supplementary script
"""

import os
import pandas as pd


def add_script_seg(data_dir):
    utt2script = {}
    supp_script = f"{data_dir}/supp/utt2script_seg"
    with open(supp_script, "r") as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip()
            uid = line.split()[0]
            text = " ".join(line.split()[1:])
            utt2script[uid] = text

    # Go through folds
    for fold_num in range(1, 13):
        for partition in ["train", "dev", "test"]:
            og_csv = f"{data_dir}/Fold_{fold_num}/{partition}_multi.csv"
            new_csv = f"{data_dir}/Fold_{fold_num}/{partition}_multi_target.csv"

            df = pd.read_csv(og_csv)

            df["script_seg"] = [utt2script[uid] for uid in df["ID"]]

            df.to_csv(new_csv, index=False)


if __name__ == "__main__":

    DATA_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

    add_script_seg(DATA_DIR)
