import time
from pymfe.mfe import MFE
from ml import read_file, read_file_openml
import pandas as pd
import sys

# dataset_name = "glass1.dat"
# read_file(sys.path[0] + "/input/" + dataset_name, "")

df, dataset_name = read_file_openml(450)

X = df.iloc[: , :-1]
y = df.iloc[: , -1:]


# model_groups = MFE.valid_groups()
# print(model_groups)

# mtfs_all = MFE.valid_metafeatures()
# print(mtfs_all)

# mtfs_subset = MFE.valid_metafeatures(groups=["general", "relative"])
# print(mtfs_subset)

# summaries = MFE.valid_summary()
# print(summaries)

start_time = time.time()
 
#groups="all", summary="all"
#exclude - clustering, info-theory
mfe = MFE(random_state=42, 
          groups=["complexity", "concept", "general", "itemset", "landmarking", "model-based", "statistical"], 
          summary=["max", "min", "median", "mean", "var", "sd", "kurtosis","skewness"])

mfe.fit(X.values, y.values)
ft = mfe.extract() #cat_cols='auto', suppress_warnings=True

finish_time = round(time.time() - start_time,3)

# print(ft)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

print("")
print("number of meta-features  :", len(ft[0]))
print("time                     :", finish_time)

#print(ft)
#print(type(ft))
#tuple

df = pd.DataFrame.from_records(ft)

new_header = df.iloc[0]
df = df[1:]
df.columns = new_header

print(df)

# df.to_csv(sys.path[0] + "/output/" + "kb_characteristics_pymfe.csv", sep=",", index=False)
