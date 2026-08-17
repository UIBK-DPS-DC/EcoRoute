import pandas as pd
from glob import glob
import random

# Combines all annotated datasets in output-14B to a single pickle file

datasets = ["cnn_dailymail", "xsum", "glue_mnli", "glue_qqp", "glue_sst2","squad","mbpp","gsm8k","natural_questions","bbh"]

root_file = "../output-14B/14B-dataset_with_step_counts_"

dfs_total = []
for ds in datasets:

    files = glob(root_file + ds + "*-*.pkl")
    df_total = None
    dfs = []
    for file in sorted(files):
        df_temp = pd.read_pickle(file)
        dfs.append(df_temp)

        print(file)
    df = pd.concat(dfs)
    dfs_total.append(df)
    df.to_pickle(f"{root_file}{ds}_comb.pkl")
    print(df)
    print("="*100)

df_total = pd.concat(dfs_total)
df_total.to_pickle("../output-14B/combined_dataset_with_step_counts.pkl")
