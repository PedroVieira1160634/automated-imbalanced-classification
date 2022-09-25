#test TOPT

from ml import read_file, read_file_openml
import sys
import time
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from tpot import TPOTClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, make_scorer, cohen_kappa_score
from imblearn.metrics import geometric_mean_score

print('\n\n----------------------------------start -', datetime.datetime.now(), '--------------------------------------\n\n')

# df, dataset_name = read_file(sys.path[0] + "/input/" + "kr-vs-k-zero_vs_eight.dat")
df, dataset_name = read_file_openml(40713)

print("dataset: ", dataset_name)

X = df.iloc[: , :-1]
y = df.iloc[: , -1:]

encoded_columns = []
for column_name in X.columns:
    if X[column_name].dtype == object or X[column_name].dtype.name == 'category' or X[column_name].dtype == bool or X[column_name].dtype == str:
        encoded_columns.extend([column_name])

if encoded_columns:
    X = pd.get_dummies(X, columns=X[encoded_columns].columns, drop_first=True)

encoded_columns = []
preserve_name = ""
for column_name in y.columns:
    if y[column_name].dtype == object or y[column_name].dtype.name == 'category' or y[column_name].dtype == bool or y[column_name].dtype == str:
        encoded_columns.extend([column_name])
        preserve_name = column_name

if encoded_columns:
    y = pd.get_dummies(y, columns=y[encoded_columns].columns, drop_first=True)

if preserve_name:
    y.rename(columns={y.columns[0]: preserve_name}, inplace = True)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) #test_size=0.25

start_time = time.time()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# generations=100, population_size=100
tpot = TPOTClassifier(generations=2, population_size=5, max_time_mins=10, scoring='f1', cv=cv, n_jobs=-1, random_state=42, verbosity=2)

model = tpot.fit(X_train, y_train.values.ravel())

# tpot.export('tpot_exported_pipeline.py')

# print("score    : %.3f" % tpot.score(X_test, y_test.values.ravel()))

model_pred = model.predict(X_test)

balanced_accuracy = round(balanced_accuracy_score(y_test, model_pred),3)
f1_score = round(f1_score(y_test, model_pred),3)
roc_auc_score = round(roc_auc_score(y_test, model_pred),3)
g_mean_score = round(geometric_mean_score(y_test, model_pred),3)
cohen_kappa = round(cohen_kappa_score(y_test, model_pred),3)

print("\nbalanced accuracy :", balanced_accuracy)
print("f1 score :", f1_score)
print("roc auc  :", roc_auc_score)
print("geometric mean score :", g_mean_score)
print("cohen kappa  :", cohen_kappa)

final_score = round(np.mean([balanced_accuracy,f1_score,roc_auc_score,g_mean_score,cohen_kappa]),3)
print("\nfinal score  :", final_score)


finish_time = round(time.time() - start_time, 3)
print("\ntime (s)       :", finish_time)
finish_time_fmt = str(datetime.timedelta(seconds=round(finish_time,0)))
print("time (HH:mm:ss):", finish_time_fmt)


# print("class name: ", tpot.__class__.__name__)
classifier = str(tpot._optimized_pipeline)
classifier = classifier.split("(", 1)[0]
print("\nclassifier: ", classifier)



df_tpot = pd.read_csv(sys.path[0] + "/output/" + "results_TPOT.csv", sep=",")

df_tpot2 = df_tpot.loc[df_tpot['dataset'] == dataset_name]

if df_tpot2.empty :

    df_tpot.loc[len(df_tpot.index)] = [
        dataset_name,
        classifier,
        finish_time,
        balanced_accuracy,
        f1_score,
        roc_auc_score,
        g_mean_score,
        cohen_kappa,
        final_score
    ]

    df_tpot.to_csv(sys.path[0] + "/output/" + "results_TPOT.csv", sep=",", index=False)
    
    print("\nTPOT Results written, rows added!","\n")

else:
    print("\nTPOT Results not written!","\n")



print('\n\n----------------------------------finish -', datetime.datetime.now(), '--------------------------------------\n\n')
