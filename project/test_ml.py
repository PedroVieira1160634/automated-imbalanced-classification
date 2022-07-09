import sys
from datetime import datetime
from ml import execute_ml

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
#car-good.dat

#pc2 (id:1069)
#mc1 (id:1056)

#get 4 random datasets


dataset_name = "page-blocks0.dat"
dataset_name = execute_ml(sys.path[0] + "/input/" + dataset_name, "")

# list_openml = [1056,316,40900,40713]
# for id in list_openml:
#     dataset_name = execute_ml("", id)
#     print("\n----------------------------------\n")


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
