import sys
import time
from datetime import datetime
from ml import execute_ml

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

start_time = time.time()

#glass1.dat                             - 2.719 sec     - 23.438 sec
#page-blocks0.dat                       - 12.868 sec    - 114.874 sec
#car-good.dat                                           - 37.169 sec
#analcatdata_lawsuit (450)                              - 28.386 sec
dataset_name = "car-good.dat"

execute_ml(sys.path[0] + "/input/" + dataset_name)
#execute_ml(450)

finish_time = (round(time.time() - start_time,3))
print("elapsed time         :", finish_time)

print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
