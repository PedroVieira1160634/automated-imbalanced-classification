print('start\n')

import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

start_time = time.time()

df = pd.read_csv(sys.path[0] + "/input/creditcard.csv")

Y = df.Class
X = df.drop('Class', axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

algorithm = XGBClassifier().fit(x_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))

print('finish')
