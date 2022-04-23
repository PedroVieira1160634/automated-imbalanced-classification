from datetime import datetime
print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

from ml import execute_ml
import sys
import time

start_time = time.time()

#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
dataset_name = "page-blocks0.dat"

execute_ml(sys.path[0] + "/input/" + dataset_name)

finish_time = (round(time.time() - start_time,3))
print("elapsed time:", finish_time)


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
