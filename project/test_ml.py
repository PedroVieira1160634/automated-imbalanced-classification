import sys
import time
from datetime import datetime
from ml import execute_ml, write_results_elapsed_time

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

#glass1.dat                         0:03:57 - 0:01:05
#page-blocks0.dat                   0:25:05 - 0:04:55
#arsenic-male-bladder (id:947)      0:04:02 - 0:01:37
#JapaneseVowels (id:976)            0:46:40 - 0:10:04   ?
#pc2 (id:1069)                              - 0:08:55   ?
#mc1 (id:1056)                              - 0:13:58   ?
#yeast_ml8 (id:316)                         - 0:14:46   bad?
#Satellite (id:40900)                       - 0:06:52   bad?
#dis (id:40713)                             - 0:03:46   ?

# dataset_name = "page-blocks0.dat"
# dataset_name = execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# list_openml = [1056,316,40900,40713]
# for id in list_openml:
#     dataset_name = execute_ml("", id)
#     print("\n----------------------------------\n")


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
