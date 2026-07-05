#
# ██████╗  ██████╗ ███████╗███████╗     ██╗
# ██╔══██╗██╔═══██╗██╔════╝██╔════╝     ██║
# ██║  ██║██║   ██║███████╗█████╗       ██║
# ██║  ██║██║   ██║╚════██║██╔══╝       ██║
# ██████╔╝╚██████╔╝███████║███████╗     ██║
# ╚═════╝  ╚═════╝ ╚══════╝╚══════╝     ╚═╝
#
# A Project by
# SAFE-AI Consortium | UNIVERSITÄTSMEDIZIN HALLE
# https://safe-ai-research.github.io
#
# Code Author: Quang Vu Nguyen
# Licence: GNU GPLv3
#


import re
from pathlib import Path
import helper_functions as hf
import pandas as pd
import statistics


def rolling_window_mean(data: pd.DataFrame, windowsize=10):
    df = []
    for i in range(0, data.shape[0]):
        s = max(0, i - windowsize)
        sub = data.iloc[range(s, i + 1)]
        mean = sub.mean(axis=0, numeric_only=True, skipna=True)
        df.append(mean)
    df = pd.DataFrame(df)
    df.columns = [f"meanlast{windowsize}s_{col}" for col in df.columns]
    return df


def intersperse(data1, data2, data3):
    if data1.shape[1] != data2.shape[1] or data1.shape[1] != data3.shape[1]:
        return None
    all = pd.DataFrame()
    names = []
    for i in range(data1.shape[1]):
        all = pd.concat([all, (pd.concat([data1.iloc[:, i], data2.iloc[:, i], data3.iloc[:, i]], axis=1))], axis=1)
        names.extend([data1.columns[i], data2.columns[i], data3.columns[i]])
    all = pd.DataFrame(all)
    all.columns = names
    return all

pathFolder = "pEEG"
pathoutput = "timelag+rollingAvg+Step4s"
folder = Path(pathFolder)
outputfolder = Path(pathoutput) # path where the files land
all_files = folder.glob("*.csv")
ids = [file.stem for file in all_files]
ids.remove('Klin_Parameter')
klin_file = pd.read_csv(pathFolder + "/" + "Klin_Parameter" + ".csv", sep="\t")

deprecated_features = ["WSMF_Klimpel", "MF_Jordan", "SEF95_Jordan", "WSMF30_16", "WSMF49_16", "WSMF_Klimpel_16",
                       "Propofol", "Endoscopy"]
target_features = ["SOC", "MOAAS"]
for file_id in ids:
    all = pd.read_csv(pathFolder + "/" + file_id + ".csv")
    all.drop(deprecated_features, axis=1, inplace=True)
    all.drop("Time", axis=1, inplace=True)
    for i in range(all.shape[1]):
        median = all.iloc[:, i].median(numeric_only=True, skipna=True)
        all.iloc[:, i] = all.iloc[:, i].fillna(median)
    data = all.drop(target_features, axis=1)
    aug1 = rolling_window_mean(data, windowsize=10)
    aug2 = rolling_window_mean(data, windowsize=60)
    data = intersperse(data, aug1, aug2)

    targetdf = all.loc[:, target_features]
    targetdf["MOAAS"] = targetdf["MOAAS"] / 5

    extracted = re.sub(r".*-(\d{3})_.*", r"\1", file_id)
    klin_row = klin_file[klin_file.iloc[:, 0] == int(extracted)]
    klin_row = klin_row.drop("STUDYID", axis=1)
    klin_data = pd.concat([klin_row] * data.shape[0], ignore_index=True)
    combined = pd.concat([targetdf, klin_data, data], axis=1)
    combined, _ = hf.create_rolling_windows(combined, window_seconds=5, window_step=4)
    f = open(f"./{outputfolder.name}/colnames.txt", "w")
    for col in combined.columns:
        f.write(f"{col}\n")
    f.close()

    combined.to_csv(f"./{outputfolder.name}/{extracted}.csv", index=False, header=False, sep=",", quoting=3)

