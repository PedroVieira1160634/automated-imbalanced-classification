from datetime import datetime
print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

from ml import *

#glass1.dat
#page-blocks0.dat
dataset_name = "page-blocks0.dat"

df = read_file(sys.path[0] + "/input/" + dataset_name)

x_train, x_test, y_train, y_test = train_test_split_func(df)

classify_evaluate_write(x_train, x_test, y_train, y_test, dataset_name)

print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
