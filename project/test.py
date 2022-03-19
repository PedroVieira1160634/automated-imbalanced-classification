print('start\n')

import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

start_time = time.time()

#glass1
#page-blocks0
df = pd.read_csv(sys.path[0] + "/input/page-blocks0.dat")

Y = df.iloc[:,-1:]
X = df.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#lgb.LGBMClassifier()
#XGBClassifier()
#RandomForestClassifier()
algorithm = RandomForestClassifier().fit(x_train, y_train)

algorithm_pred = algorithm.predict(x_test)

print("\n-------")

print('accuracy:')
print(round(accuracy_score(y_test, algorithm_pred),5),"\n")

print('f1 score:')
print(round(f1_score(y_test, algorithm_pred),5),"\n")

print('roc_auc_score:')
print(round(roc_auc_score(y_test, algorithm_pred),5),"\n")

print("%s seconds" % (round(time.time() - start_time,5)))

print('\nfinish')
