import sys
import time
import datetime
import pandas as pd
import numpy as np
from ml import get_best_results_by_characteristics, execute_byCharacteristics, write_characteristics_remove_current_dataset, display_final_results

# a = (1, 2, 3)
# b = (4, 5, 6)
# dst = np.linalg.norm(np.array(a) - np.array(b))
# print(dst)


# dataset_name = ""
# print("\ndataset selected:", dataset_name, "\n")
# df_dist = get_best_results_by_characteristics(dataset_name)
# print(df_dist)

# df_dist = pd.read_csv(sys.path[0] + "/input/" + "test_ui.csv", sep=",")
# str_output = display_final_results(df_dist)
# print(str_output)


# dataset_name = ""
# str_output = execute_byCharacteristics(sys.path[0] + "/input/" + dataset_name, "")
# print(str_output)

#40713, 1116

start_time = time.time()

str_output = execute_byCharacteristics("", 40713)
# print("\nFinal UI Table:\n", str_output)

finish_time = (round(time.time() - start_time,3))
finish_time = str(datetime.timedelta(seconds=round(finish_time,0)))
print("Elapsed Time                 :", finish_time, "\n")

