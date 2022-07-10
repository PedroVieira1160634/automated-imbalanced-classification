import sys
from datetime import datetime
from ml import execute_ml

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n')

#glass1.dat
#page-blocks0.dat
#car-good.dat
#pc2 (id:1069)
#mc1 (id:1056)

#get 5 random datasets


# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

execute_ml("", 1056)

# list_openml = [1056,316,40900,40713]
# for id in list_openml:
#     execute_ml("", id)
#     print("\n----------------------------------\n")


print('\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
