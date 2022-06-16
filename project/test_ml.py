import sys
import time
from datetime import datetime
from ml import execute_ml, write_results_elapsed_time

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

start_time = time.time()

#glass1.dat
#page-blocks0.dat
#arsenic-male-bladder (id:947)
#JapaneseVowels (id:976)
#pc2 (id:1069)

# dataset_name = "glass1.dat"
# dataset_name = execute_ml(sys.path[0] + "/input/" + dataset_name, "")

dataset_name = execute_ml("", 976)


finish_time = (round(time.time() - start_time,3))
# print("Elapsed Time         :", finish_time)

if dataset_name:
    write_results_elapsed_time(finish_time, dataset_name)


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
