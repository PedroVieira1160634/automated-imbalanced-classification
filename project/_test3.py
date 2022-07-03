import sys
import pandas as pd
import numpy as np
from ml import get_best_results_by_characteristics, execute_byCharacteristics, write_characteristics_remove_current_dataset

# a = (1, 2, 3)
# b = (4, 5, 6)
# dst = np.linalg.norm(np.array(a) - np.array(b))
# print(dst)


# dataset_name = "kddcup-rootkit-imap_vs_back.dat"
# print("\ndataset selected:", dataset_name, "\n")
# df_dist = get_best_results_by_characteristics(dataset_name)
# print(df_dist)

dataset_name = "kddcup-rootkit-imap_vs_back.dat"
df_dist = execute_byCharacteristics(sys.path[0] + "/input/" + dataset_name, "")
print(df_dist)

# write_characteristics_remove_current_dataset()

