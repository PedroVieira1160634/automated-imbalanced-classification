from datetime import datetime
print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

import sys
import time
from ml import execute_ml

start_time = time.time()

#glass1.dat                             - 2.719 sec     - 27.891 sec
#page-blocks0.dat                       - 12.868 sec    - 128.502 sec
#car-good.dat
#kddcup-rootkit-imap_vs_back.dat        - 3.435 sec     - 36.234 sec
dataset_name = "kddcup-rootkit-imap_vs_back.dat"

execute_ml(sys.path[0] + "/input/" + dataset_name)

finish_time = (round(time.time() - start_time,3))
print("elapsed time:", finish_time)


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
