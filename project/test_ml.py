import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# execute_ml("", 41538)

# list_openml = [1487,40713,1116] #41538,4329,1447,949,951,1451,1487,40713,1116
# for id in list_openml:
#     execute_ml("", id)
#     print("\n----------------------------------\n")


print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')

#excel results - execute_ml_test
