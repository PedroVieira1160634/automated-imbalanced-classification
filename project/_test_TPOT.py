#test TOPT

import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datetime import datetime
print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

#glass1.dat
#page-blocks0.dat
df = pd.read_csv(sys.path[0] + "/input/" + "glass1.dat")

X = df.iloc[:,:-1]
y = df.iloc[:,-1:]

encoded_columns = []
for column_name in X.columns:
    if X[column_name].dtype == object:
        encoded_columns.extend([column_name])
    else:
        pass

X = pd.get_dummies(X, X[encoded_columns].columns, drop_first=True)

encoded_columns = []
for column_name in y.columns:
    if y[column_name].dtype == object:
        encoded_columns.extend([column_name])
    else:
        pass

y = pd.get_dummies(y, y[encoded_columns].columns, drop_first=True)


smote = SMOTE(random_state=42) #, k_neighbors=minimum_samples
X, y = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

start_time = time.time()

#https://epistasislab.github.io/tpot/api/
#generations=5, population_size=20, cv=5
algorithm = TPOTClassifier(generations=2, population_size=5, scoring='f1', cv=2, n_jobs=-1, random_state=42, verbosity=2, memory='auto')

model = algorithm.fit(X_train, y_train.values.ravel())

#print("score    : %.3f" % algorithm.score(X_test, y_test.values.ravel()))

#algorithm.export('tpot_exported_pipeline.py')

finish_time = time.time() - start_time
print("time     : %.3f" % finish_time)


model_pred = model.predict(X_test)

print("accuracy :", round(accuracy_score(y_test, model_pred),3))
print("f1 score :", round(f1_score(y_test, model_pred),3))
print("roc auc  :", round(roc_auc_score(y_test, model_pred),3))


print("")
# print("class name: ", algorithm.__class__.__name__)
class_name = str(algorithm._optimized_pipeline)
class_name = class_name.split("(", 1)[0]
print(class_name)



#glass1.dat
#without preprocessing
#F1 -   0.710
#time - 86 sec
#GradientBoostingClassifier


#glass1.dat
#SMOTE
#F1 -   0.923
#time - 97 sec
#RandomForestClassifier

print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
