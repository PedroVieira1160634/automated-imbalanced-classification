import sys
from datetime import datetime
from ml import execute_ml, execute_ml_test

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------')

# FILE

# # dataset_name = "car-good.dat"
# # execute_ml_test(sys.path[0] + "/input/" + dataset_name, "")
# # execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# list_files = ["poker-8-9_vs_6.dat","kr-vs-k-zero-one_vs_draw.dat","kr-vs-k-zero_vs_eight.dat","kr-vs-k-three_vs_eleven.dat","dermatology-6.dat",
#                "car-vgood.dat","abalone-21_vs_8.dat","kr-vs-k-zero_vs_fifteen.dat","poker-8_vs_6.dat","car-good.dat","page-blocks0.dat","glass1.dat"]
# for dataset_name in list_files:
#     execute_ml(sys.path[0] + "/input/" + dataset_name, "")
#     print("\n----------------------------------\n")

#FILE


# #OPENML

# execute_ml_test("", 986)
# execute_ml("", 986)

# list_openml = [986, 43595, 1010, 897, 13, 990]
# for id in list_openml:
#     execute_ml_test("", id)
#     print("\n----------------------------------\n")

# execute_ml("", 765)

# list_openml = [990, 897, 1010, 975, 950, 980, 767, 311, 958, 995, 40900, 765, 40994, 42172, 1003, 1037, 966, 1023, 968, 42883, 833, 761, 735, 1489,
#                1524, 1511, 312, 1004, 934, 1025, 728, 41156, 796, 994, 733, 337, 40701, 978, 971, 947, 40983, 1016, 962, 1049, 954, 976, 1022, 764,
#                1020, 41946, 450, 1116, 40713]
list_openml = [980, 767, 311, 958, 995]
for id in list_openml:
    execute_ml("", id)
    print("\n----------------------------------\n")
# #OPENML

print('----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')

#excel results - execute_ml_test
