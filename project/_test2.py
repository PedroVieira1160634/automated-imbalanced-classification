import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ml import read_file, read_file_openml

#glass1.dat page-blocks0.dat car-good.dat
dataset_name = "glass1.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

#1069 1056
# df, dataset_name = read_file_openml(1056)

print(df)
