import sys
import pandas as pd

df_openml = pd.read_csv(sys.path[0] + "/input/" + "datasets_openml.csv", sep=",")

rows_to_keep = df_openml["imbalance ratio"] > 10

df_openml = df_openml[rows_to_keep]

print(df_openml)

#450, ...
