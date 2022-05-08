#example ML

import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
df = pd.read_csv(sys.path[0] + "/input/" + "page-blocks0.dat")

X = df.iloc[:,:-1]
y = df.iloc[:,-1:]


# -- characteristics of datasets

print("")
print("numeric columns      :", X.select_dtypes(include=np.number).shape[1])
print("non-numeric columns  :", X.select_dtypes(include=object).shape[1])



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

# -- characteristics of datasets

#todo fix problems
#if y.values.tolist().count([1]) > 0
#if y.values.tolist().count([0]) > y.values.tolist().count([1])

print("imbalance ratio      : %.3f" % (y.values.tolist().count([0])/y.values.tolist().count([1])))
print("")


#print(list(df.columns))
#print(list(X.columns))
#print(list(y.columns))

#print(y.value_counts())


smote = SMOTE(random_state=42) #, k_neighbors=minimum_samples
X, y = smote.fit_resample(X, y)

#print(y.value_counts())

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#0.452 sec (23x more fast)

start_time = time.time()

algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced') #.fit(X_train, y_train.values.ravel())
#algorithm = KNeighborsClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

scores_f1 = cross_val_score(algorithm, X, y.values.ravel(), scoring='f1', cv=cv, n_jobs=-1) #scoring='roc_auc' f1 balanced_accuracy
#10.547 sec
scores_roc_auc = cross_val_score(algorithm, X, y.values.ravel(), scoring='roc_auc', cv=cv, n_jobs=-1)
#17.509 sec (with both)

finish_time = time.time() - start_time

print('Mean F1 Score        : %.3f' % np.mean(scores_f1))
print('Mean ROC AUC Score   : %.3f' % np.mean(scores_roc_auc))

print('time             : %.3f' % finish_time)



#see if overfitting
# algorithm_pred = algorithm.predict(X_train)

# print("train")
# print("metric_accuracy:         ", round(accuracy_score(y_train, algorithm_pred),3))
# print("metric_f1_score          ", round(f1_score(y_train, algorithm_pred),3))
# print("metric_roc_auc_score:    ", round(roc_auc_score(y_train, algorithm_pred),3))

# algorithm_pred = algorithm.predict(X_test)

# print("test")
# print("metric_accuracy:         ", round(accuracy_score(y_test, algorithm_pred),3))
# print("metric_f1_score          ", round(f1_score(y_test, algorithm_pred),3))
# print("metric_roc_auc_score:    ", round(roc_auc_score(y_test, algorithm_pred),3))
# print("time:                    ", finish_time)



