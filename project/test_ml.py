import sys
import time
from datetime import datetime
from ml import execute_ml, write_results_elapsed_time

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

#glass1.dat
#page-blocks0.dat

# kddcup-rootkit-imap_vs_back.dat
# car-good.dat
# analcatdata_lawsuit (id:450)

#arsenic-male-bladder (id:947)
#JapaneseVowels (id:976)
#pc2 (id:1069)
#mc1 (id:1056)
#yeast_ml8 (id:316)
#Satellite (id:40900)
#dis (id:40713)

# dataset_name = "page-blocks0.dat"
# dataset_name = execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# list_openml = [1056,316,40900,40713]
# for id in list_openml:
#     dataset_name = execute_ml("", id)
#     print("\n----------------------------------\n")


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
