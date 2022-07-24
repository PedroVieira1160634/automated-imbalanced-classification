import sys
import pandas as pd
from ml import read_file

# df = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")
# df = df.sample(n=13)

# print(df[["dataset", "total elapsed time"]])

#excel results


# dataset_name = "car-good.dat"
# df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

#average of metrics

# y = df.iloc[: , -1:]
# imbalance_ratio = 0
# if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
#     if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
#         imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
#     else:
#         imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)

# print("imbalance ratio      :", imbalance_ratio)
# print("")


df = pd.read_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",")
df.sort_values(by=['final score'], ascending=False, inplace=True)
# df.to_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",", index=False)
