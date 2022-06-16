import pandas as pd
import openml.datasets
from ml import read_file_openml

df, dataset_name = read_file_openml(1069)

print("\ndf:")
print(df)

print("\ndataset_name:")
print(dataset_name)
