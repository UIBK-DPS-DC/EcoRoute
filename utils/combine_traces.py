import os
import re
from glob import glob

import pandas as pd

path = "../processed"

dic = {}

# filter for dataset and model
for file in glob(os.path.join(path, "processed*.pkl")):
    pattern = ".*processed-(.*)-(\d*)-(\d*)_(\d*).pkl"
    match = re.match(pattern, file)
    
    dataset = match.group(1)
    model = match.group(2)
    from_idx = int(match.group(3))
    to_idx = int(match.group(4))

    if(dataset not in dic):
        dic[dataset] = {}
    
    if(model not in dic[dataset]):
        dic[dataset][model] = []
    
    dic[dataset][model].append(file)

# sort files by row index
for k, v in dic.items():
    for k_1, v_1 in dic[k].items():
        dic[k][k_1] = sorted(v_1)

common_columns = ['task', 'prompt']
prefixes = ("response", "time", "energy", "finish_reason")

# combine to single file for each dataset
for k, v in dic.items():
    print(f"Combining traces for dataset {k}...")
    vals = v.values()

    # concatenate response, response time and energy 
    # of all models into a single dataframe
    combined_horizontal = []
    for files in zip(*vals):
        dfs = [ pd.read_pickle(file) for file in files ]
        base_cols = [col for col in dfs[0].columns if not col.startswith(prefixes)]

        # Take base columns once
        result = dfs[0][base_cols].copy()

        # Collect variable columns from all dfs
        var_parts = []
        for df in dfs:
            cols = [c for c in df.columns if c.startswith(prefixes)]
            var_parts.append(df[cols])

        # Concatenate horizontally
        result = pd.concat([result] + var_parts, axis=1)
        combined_horizontal.append(result)
    
    # stack all rows into a single dataframe
    combined_vertical = pd.concat(combined_horizontal, ignore_index=True)
    # combined_vertical = combined_vertical[:~combined_vertical.columns.duplicated()].copy()
    duplicate_cols = combined_vertical.columns[combined_vertical.columns.duplicated()]
    # combined_vertical = combined_vertical.T.drop_duplicates().T
    combined_vertical.drop(columns=duplicate_cols, inplace=True)

    filename = os.path.join(path, f"combined_{k}.pkl")
    print(f"Saving to {filename}")
    combined_vertical.to_pickle(filename)
    