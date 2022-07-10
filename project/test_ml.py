import sys
from datetime import datetime
from ml import execute_ml

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n')

#glass1.dat
#page-blocks0.dat

#car-good.dat
#pc2 (id:1069)

#40994  -0:24
#450    -0:08
#41946  -0:47

#1020   -01:31


#mc1 (id:1056)  demora muito
#43897          tem 1 metricas
#43895          tem 1 metricas




# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

execute_ml("", 1020)

# list_openml = [40994,450,41946] #1020
# for id in list_openml:
#     execute_ml("", id)
#     print("\n----------------------------------\n")


print('\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
