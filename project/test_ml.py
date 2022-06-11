import sys
import time
from datetime import datetime
from ml import execute_ml

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

start_time = time.time()

#glass1.dat                             - 2.719 sec     - 28.890 sec
#page-blocks0.dat                       - 12.868 sec    - 147.156 sec
#car-good.dat                                           - 44.085 sec
#analcatdata_lawsuit (id:450)                           - 29.272 sec    - 196.045 sec

# dataset_name = "glass1.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

execute_ml("", 450)


finish_time = (round(time.time() - start_time,3))
print("elapsed time         :", finish_time)

print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
