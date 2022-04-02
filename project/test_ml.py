from datetime import datetime
print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

from ml import *

#glass1.dat
#page-blocks0.dat
dataset_name = "page-blocks0.dat"

#-, SMOTE, OVER, UNDER
balancing = "-"

df = read_file(sys.path[0] + "/input/" + dataset_name)
x_train, x_test, y_train, y_test = train_test_split_func(df, balancing)
resultsList = classify_evaluate_write(x_train, x_test, y_train, y_test, dataset_name, balancing)
print("\n", find_best_result(resultsList), "\n")

print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
