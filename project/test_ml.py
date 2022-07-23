import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

#, "kb_results", "kb_full_results", "kb_characteristics"
#, "testing_kb_results", "testing_kb_full_results", "testing_kb_characteristics"

# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# execute_ml("", 975, "kb_results", "kb_full_results", "kb_characteristics")

list_openml = [986, 43595, 1010, 897, 13, 990]
for id in list_openml:
    execute_ml("", id, "kb_results", "kb_full_results", "kb_characteristics")
    print("\n----------------------------------\n")

#975, 986, 43595, 1010, 897, 13, 990

print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')

#excel results - execute_ml_test
