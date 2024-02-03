import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

#FILE
# dataset_name = "car-good.dat"
# execute_ml_test(sys.path[0] + "/input/" + dataset_name, "")
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")
#FILE



#OPENML
execute_ml_test("", 986) #765
# execute_ml("", 986)

# list_openml = [986, 43595, 1010, 897, 13, 990]
# for id in list_openml:
#     execute_ml("", id)
#     print("\n----------------------------------\n")
#OPENML

print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')

#excel results - execute_ml_test
