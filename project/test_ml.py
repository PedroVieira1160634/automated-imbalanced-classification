import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n')

# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# execute_ml("", 1506)

# list_openml = [767,1049,980]
# for id in list_openml:
#     execute_ml("", id)
#     print("\n----------------------------------\n")

print('\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')


#runned:

#glass1.dat
#page-blocks0.dat

#car-good.dat
#pc2 (id:1069) bad metric

#40994  0:24
#450    0:08
#41946  0:47

#1020   01:31
#1506   0:08 bad metric
#765    0:10
#43894  0:22
#40900  1:32
#1452   0:12 bad metric
#1558   1:08
#1467   0:11
#995    0:55
#43905  0:21 bad metric
#958    0:30
#1065   0:09
#1067   0:24
#764    0:06
#311    0:22
#1068   0:17
#43893  0:22
#1022   4:19
#976    7:08
#1444   0:27
#954    0:29
#1050   0:29
#1446   0:08
#767    0:05
#1049   0:22
#980    2:00
