import time
from pymfe.mfe import MFE
from ml import read_file, read_file_openml, write_characteristics
import pandas as pd
import sys
# import warnings
# warnings.filterwarnings("ignore")

dataset_name = "glass1.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

# df, dataset_name = read_file_openml(450) #450 1069

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
          summary=["mean", "sd", "kurtosis","skewness"]) #"max", "min", "median", "var"

mfe.fit(X.values, y.values)
ft = mfe.extract() #cat_cols='auto', suppress_warnings=True

finish_time = round(time.time() - start_time,3)

# print(ft)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

print("")
print("number of meta-features  :", len(ft[0]))
print("time                     :", finish_time)
print("")

#print(ft)
#print(type(ft))

df_characteristics = pd.DataFrame.from_records(ft)

new_header = df_characteristics.iloc[0]
df_characteristics = df_characteristics[1:]
df_characteristics.columns = new_header

#dataset
df_characteristics.insert(loc=0, column="dataset", value=[dataset_name])

print(df_characteristics)

# df_characteristics.to_csv(sys.path[0] + "/output/" + "kb_characteristics_pymfe.csv", sep=",", index=False)

# print(df_characteristics["dataset"] == "glass1.dat")
# print(len(df_characteristics.columns))
