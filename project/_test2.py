import sys
import pandas as pd
import numpy as np
from decimal import Decimal
from ml import read_file, features_labels, write_characteristics

#glass1.dat
#page-blocks0.dat
#car-good.dat
dataset_name = "glass1.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

X, y, characteristics = features_labels(df, dataset_name)

write_characteristics(characteristics)

