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


#41538   0:01       conference_attendance
#4329    0:01       thoracic_surgery
#1447    0:01       CastMetal1
#949     0:01       arsenic-female-bladder
#951     0:02       arsenic-male-lung
#1451    0:02       PieChart1 
#1487    0:18       ozone-level-8hr
#40713   0:24       dis
#1116    4:13       musk
#43051   too long?  art_daily_nojump4

#see also results

start_time = time.time()

str_output = execute_byCharacteristics("", 949)
print("\nFinal UI Table:\n", str_output)

finish_time = (round(time.time() - start_time,3))
finish_time = str(datetime.timedelta(seconds=round(finish_time,0)))
print("Elapsed Time                 :", finish_time, "\n")

