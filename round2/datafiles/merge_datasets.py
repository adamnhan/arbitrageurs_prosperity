import pandas as pd
import os

file_paths = [
    "datasets/prices_round_2_day_-1.csv",
    "datasets/prices_round_2_day_0.csv",
    "datasets/prices_round_2_day_1.csv"
]

dataframes = []
max_timestamp_so_far = 0

for file_path in sorted(file_paths, key=lambda x: int(os.path.basename(x).split("_day_")[1].split(".csv")[0])):
    day = int(os.path.basename(file_path).split("_day_")[1].split(".csv")[0])
    df = pd.read_csv(file_path, delimiter=";")  # use the correct delimiter
    df["day"] = day
    
    # Shift timestamps to maintain a continuous timeline
    df["timestamp"] += max_timestamp_so_far
    
    # Update max_timestamp_so_far
    max_timestamp_so_far = df["timestamp"].max() + 1

    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)
merged_df.to_csv("merged_data_continuous.csv", index=False)
print("done")
