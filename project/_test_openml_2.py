import sys
import pandas as pd

df_openml = pd.read_csv(sys.path[0] + "/input/" + "datasets_openml.csv", sep=",")

rows_to_keep = df_openml["imbalance ratio"] > 5

# imbalance ratio > 10 -> 22 datasets
# imbalance ratio > 5  -> 63 datasets

df_openml = df_openml[rows_to_keep]

df_openml.sort_values(by=['imbalance ratio'], inplace=True)

print(df_openml)

print("\ncount:")
print(df_openml.count())

#976 ... 1069
