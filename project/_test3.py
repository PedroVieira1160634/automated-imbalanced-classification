import sys
import pandas as pd
import numpy as np
from ml import get_best_results_by_characteristics, execute_byCharacteristics, write_characteristics_remove_current_dataset

# a = (1, 2, 3)
# b = (4, 5, 6)
# dst = np.linalg.norm(np.array(a) - np.array(b))
# print(dst)


# # dataset_name = "kddcup-rootkit-imap_vs_back.dat"
# # print("\ndataset selected:", dataset_name, "\n")
# # df_dist = get_best_results_by_characteristics(dataset_name)
# # print(df_dist)

# dataset_name = "kddcup-rootkit-imap_vs_back.dat"
# df_dist = execute_byCharacteristics(sys.path[0] + "/input/" + dataset_name, "")
# print(df_dist)

# # write_characteristics_remove_current_dataset()

df_dist = pd.read_csv(sys.path[0] + "/input/" + "test_ui.csv", sep=",")

df_dist.loc[-1] = ['Pre Processing', 'Algorithm']
df_dist.index = df_dist.index + 1
df_dist = df_dist.sort_index()

# list_dist = df_dist.values.tolist()
# print(list_dist)
# print(df_dist['pre processing'])
# print(df_dist)
str_output = "\n".join("{:30} {:30}".format(x, y) for x, y in zip(df_dist['pre processing'], df_dist['algorithm']))
# str_output = "\n".join("{:30} {:30}".format(x, y) for x, y in zip(df_dist['pre processing'], df_dist['algorithm']))
print(str_output)

# print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(df_dist['pre processing'], df_dist['algorithm'])))

