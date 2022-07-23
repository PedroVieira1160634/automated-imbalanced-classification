import sys
import pandas as pd

df = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")
df = df.sample(n=13)
print(df[["dataset", "c2"]])




