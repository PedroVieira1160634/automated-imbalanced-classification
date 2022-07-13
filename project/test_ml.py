import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# execute_ml("", 43896)

list_openml = [316,40910]
for id in list_openml:
    execute_ml("", id)
    print("\n----------------------------------\n")


print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')


#runned:

#glass1.dat
#page-blocks0.dat

#car-good.dat
#1069   2:02    bad metrics

#40994  0:24
#450    0:08
#41946  0:47

#1020   1:31
#1506   0:08    bad metrics
#765    0:10
#43894  0:22
#40900  1:32
#1452   0:12    bad metrics
#1558   1:08
#1467   0:11
#995    0:55
#43905  0:21    bad metrics
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

#43896  0:22
#1412   0:05
#1453   0:17
#1449   0:04
#1442   0:05
#962    0:10
#1016   0:09
#1443   0:10
#1071   0:06
#40983  0:59
#947    0:05
#950    0:05
#971    1:15
#978    2:36
#40701  1:52
#
#316    2:28    bad metric?
#40910  19:31   too long! bad metric!
