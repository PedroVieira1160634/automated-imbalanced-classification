import sys
import pandas as pd
from ml import read_file, read_file_openml

# df = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")
# df = df.sample(n=13)

# print(df[["dataset", "total elapsed time"]])

#excel results


# dataset_name = "kr-vs-k-zero_vs_eight.dat"
# df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

df, dataset_name = read_file_openml(976)

y = df.iloc[: , -1:]

encoded_columns = []
preserve_name = ""
for column_name in y.columns:
    if y[column_name].dtype == object or y[column_name].dtype.name == 'category' or y[column_name].dtype == bool or y[column_name].dtype == str:
        encoded_columns.extend([column_name])
        preserve_name = column_name

if encoded_columns:
    y = pd.get_dummies(y, columns=y[encoded_columns].columns, drop_first=True)

if preserve_name:
    y.rename(columns={y.columns[0]: preserve_name}, inplace = True)
    
imbalance_ratio = 0
if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
    if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
        imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
    else:
        imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)

print("imbalance ratio      :", imbalance_ratio)
print("")



# df = pd.read_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",")
# df.sort_values(by=['final score'], ascending=False, inplace=True)
# df.to_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",", index=False)
# print(df)
