import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

# dataset_name = "car-good.dat"
# execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# execute_ml("", 1056)

list_openml = [40704,1511,1524,833,735,761,31,1489,23499,1480,41945]
for id in list_openml:
    execute_ml("", id)
    print("\n----------------------------------\n")


print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')


#runned:

#glass1.dat
#page-blocks0.dat

#car-good.dat

#40994  0:24
#450    0:08
#41946  0:47

#1020   1:31
#765    0:10
#40900  1:32
#1558   1:08
#995    0:55
#958    0:30
#1065   0:09
#1067   0:24
#764    0:06
#311    0:22
#1068   0:17
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
#316    2:28
#40910  19:31

#337    0:10
#733    0:02
#994    0:06
#796    0:03
#41156  0:55
#728    0:44
#1025   0:03
#934    0:09
#1004   0:16
#312    8:11
#
#40704  0:11        Titanic
#1511   0:04        wholesale-customers
#1524   0:03        vertebra-column
#833    7:01        bank32nh
#735    3:55        cpu_small
#761    4:17        cpu_act
#31     0:06        credit-g
#1489   1:17        phoneme
#23499  0:03        breast-cancer-dropped-missing-attributes-values
#1480   0:04        ilpd
#41945  0:05        ilpd-numeric
