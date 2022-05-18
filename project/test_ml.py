import sys
import time
from ml import execute_ml
from datetime import datetime
print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

start_time = time.time()

#glass1.dat                             - 2.719 sec     - 34.246 sec
#page-blocks0.dat                       - 12.868 sec    - 146.825 sec
#car-good.dat                                           - 36.940 sec
dataset_name = "car-good.dat"

execute_ml(sys.path[0] + "/input/" + dataset_name)

finish_time = (round(time.time() - start_time,3))
print("elapsed time:", finish_time)


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
