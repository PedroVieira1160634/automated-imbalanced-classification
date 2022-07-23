import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

#, "kb_results", "kb_full_results", "kb_characteristics"
#, "testing_kb_results", "testing_kb_full_results", "testing_kb_characteristics"

#FILE
dataset_name = "kddcup-land_vs_satan.dat"

# execute_ml_test(sys.path[0] + "/input/" + dataset_name, "")
execute_ml(sys.path[0] + "/input/" + dataset_name, "", "kb_results", "kb_full_results", "kb_characteristics")
#FILE


#OPENML
#testing
# execute_ml_test("", 975)

# list_openml = [986, 43595, 1010, 897, 13, 990]
# for id in list_openml:
#     execute_ml_test("", id)
#     print("\n----------------------------------\n")

# execute_ml("", 975, "kb_results", "kb_full_results", "kb_characteristics")

# list_openml = [986, 43595, 1010, 897, 13, 990]
# for id in list_openml:
#     execute_ml("", id, "kb_results", "kb_full_results", "kb_characteristics")
#     print("\n----------------------------------\n")
#OPENML

print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')

#excel results - execute_ml_test
