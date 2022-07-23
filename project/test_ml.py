import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

#, "testing_kb_results", "testing_kb_full_results", "testing_kb_characteristics"

# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# execute_ml("", 975)

list_openml = [975, 986, 43595, 1010, 897, 13, 990]
for id in list_openml:
    execute_ml_test("", id)
    print("\n----------------------------------\n")

#975, 986, 43595, 1010, 897, 13, 990

print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')

#excel results - execute_ml_test
