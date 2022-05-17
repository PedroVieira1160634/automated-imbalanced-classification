#example ML

from datetime import datetime
print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')


import sys
import time
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
#car-good.dat
df = pd.read_csv(sys.path[0] + "/input/" + "car-good.dat")

X = df.iloc[:,:-1]
y = df.iloc[:,-1:]


# -- characteristics of datasets

n_columns = len(X.columns)
n_numeric_col = X.select_dtypes(include=np.number).shape[1]
n_non_numeric_col = X.select_dtypes(include=object).shape[1]


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

print("number of rows       :", len(df))
print("number of columns    :", n_columns)
print("numeric columns      :", n_numeric_col)
print("non-numeric columns  :", n_non_numeric_col)
print("")


# X2 = copy.deepcopy(X)
# X2['Class'] = y
c = df.corr().abs() #ignore categorical columns
sol = (c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=False))

print("max correlation      : %.3f" % (sol.max()))
print("average correlation  : %.3f" % (sol.mean(skipna = True)))
print("minimum correlation  : %.3f" % (sol.min()))
print("")


print("distinct values in columns:")
for column in df:
    print(column, " - ", df[column].nunique())
print("")

list_unique_values = []
for column in df:
    list_unique_values.append(df[column].nunique())
print("average of distinct values in columns: %.0f" % (np.mean(list_unique_values)))
print("")


imbalance_ratio = 0
if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
    if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
        imbalance_ratio = y.values.tolist().count([0])/y.values.tolist().count([1])
    else:
        imbalance_ratio = y.values.tolist().count([1])/y.values.tolist().count([0])

print("imbalance ratio      : %.3f" % (imbalance_ratio))
print("")




#print(y.value_counts())

smote = SMOTE(random_state=42) #, k_neighbors=minimum_samples
X, y = smote.fit_resample(X, y)

#print(y.value_counts())



print("\nFinal Results:\n")

#   --- k-Fold Cross-Validation ---

start_time = time.time()

algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced') #.fit(X_train, y_train.values.ravel())
#algorithm = KNeighborsClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

scoring = {'balanced_accuracy': 'balanced_accuracy',
           'f1': 'f1', 
           'roc_auc': 'roc_auc'}

#, return_train_score=True
scores = cross_validate(algorithm, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1)
#10.837

finish_time = time.time() - start_time

#print("Mean F1 Score        : %.3f" % np.mean(scores_f1))
#print("Mean ROC AUC Score   : %.3f" % np.mean(scores_roc_auc))

# print("train:")
# print("Mean Accuracy Score  : %.3f" % np.mean(scores['train_balanced_accuracy']))
# print("Mean F1 Score        : %.3f" % np.mean(scores['train_f1']))
# print("Mean ROC AUC         : %.3f" % np.mean(scores['train_roc_auc']))
# print("")

# print("test:")
print("Mean Accuracy Score  : %.3f" % np.mean(scores['test_balanced_accuracy']))
print("Mean F1 Score        : %.3f" % np.mean(scores['test_f1']))
print("Mean ROC AUC         : %.3f" % np.mean(scores['test_roc_auc']))

print("")
print("time                 : %.3f" % finish_time)

#   --- k-Fold Cross-Validation ---




#   --- train/test ---

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #0.452 sec (23x more fast)

# start_time = time.time()

# algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced').fit(X_train, y_train.values.ravel())

# finish_time = time.time() - start_time

# #see if overfitting
# # algorithm_pred = algorithm.predict(X_train)

# # print("train")
# # print("metric_accuracy:         ", round(accuracy_score(y_train, algorithm_pred),3))
# # print("metric_f1_score          ", round(f1_score(y_train, algorithm_pred),3))
# # print("metric_roc_auc_score:    ", round(roc_auc_score(y_train, algorithm_pred),3))

# algorithm_pred = algorithm.predict(X_test)

# #print("test")
# print("metric_accuracy:         ", round(accuracy_score(y_test, algorithm_pred),3))
# print("metric_f1_score:         ", round(f1_score(y_test, algorithm_pred),3))
# print("metric_roc_auc_score:    ", round(roc_auc_score(y_test, algorithm_pred),3))

# print("")
# print("time:                    ", round(finish_time,3))

#   --- train/test ---


print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')

