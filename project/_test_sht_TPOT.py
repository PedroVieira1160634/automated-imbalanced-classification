#version of TPOT: Version 0.11.7

from ml import read_file, read_file_openml
import sys
import time
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from tpot import TPOTClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, make_scorer, cohen_kappa_score
from imblearn.metrics import geometric_mean_score
from ml import *

print('\n\n----------------------------------start -', datetime.datetime.now(), '--------------------------------------\n\n')

class ResultsSHT(object):
    def __init__(self, dataset_name, algorithm, output_app, time, final_score_values, final_score_std_values):
        self.dataset_name = dataset_name
        # self.balancing = balancing
        self.algorithm = algorithm
        self.output_app = output_app
        self.time = time
        self.final_score_values = final_score_values
        self.final_score_std_values = final_score_std_values
        
def write_results_sht(result):
    if not result:
        print("--result not valid on write_results--")
        print("result:", result)
        return False
    
    try:
        
        df_kb_r = pd.read_csv(application_path + "/output/" + "results_sht.csv", sep=",")
        
        df_kb_r2 = df_kb_r.loc[(df_kb_r['dataset'] == result.dataset_name) & (df_kb_r['output app'] == result.output_app)]
        
        if df_kb_r2.empty :
        
            fold = 1
            for final_score, final_score_std in zip(result.final_score_values, result.final_score_std_values):
                df_kb_r.loc[len(df_kb_r.index)] = [
                    result.dataset_name,
                    '-', # result.balancing,
                    result.algorithm,
                    result.output_app,
                    fold,
                    result.time,
                    final_score,
                    final_score_std
                ]
                fold += 1

            df_kb_r.to_csv(application_path + "/output/" + "results_sht.csv", sep=",", index=False)
            
            print("Results SHT written, row added!","\n")
        
        else:
            print("Results SHT not written!","\n")
        
    except Exception:
        traceback.print_exc()
        return False
    
    return True


#MUDAR AQUI
# df, dataset_name = read_file(sys.path[0] + "/input/" + "kr-vs-k-zero_vs_eight.dat") #"page-blocks0.dat
df, dataset_name = read_file_openml(976)

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

scoring = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score, greater_is_better=True),
    'f1': 'f1', 
    'roc_auc': make_scorer(roc_auc_score, greater_is_better=True),
    'g_mean': make_scorer(geometric_mean_score, greater_is_better=True),
    'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True)
    }


# generations=100, population_size=100
#scoring='f1', cv=cv
tpot = TPOTClassifier(generations=2, population_size=2, max_time_mins=10, scoring=scoring, cv=cv, n_jobs=-1, random_state=42, verbosity=2, disable_update_check=True)


model = tpot.fit(X_train, y_train.values.ravel())

# scores = cross_validate(tpot, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1)


# tpot.export('tpot_exported_pipeline.py')

# print("score    : %.3f" % tpot.score(X_test, y_test.values.ravel()))

# model_pred = model.predict(X_test)

finish_time = round(time.time() - start_time, 3)




best_pipeline = tpot.fitted_pipeline_
# print("\nbest_pipeline: ", best_pipeline)

scores = cross_validate(best_pipeline, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1)
# print("\nscores: ", scores)







# print("scores: ", scores)

balanced_accuracy_scores = scores['test_balanced_accuracy']
f1_score_scores = scores['test_f1']
roc_auc_score_scores = scores['test_roc_auc']
g_mean_score_scores = scores['test_g_mean']
cohen_kappa_scores = scores['test_cohen_kappa']

final_score_values = []
final_score_std_values = []

for i, (balanced_accuracy_value, f1_score_value, roc_auc_score_value, g_mean_score_value, cohen_kappa_value) in enumerate(zip(balanced_accuracy_scores, f1_score_scores, roc_auc_score_scores, g_mean_score_scores, cohen_kappa_scores)):
    final_score_value = round(np.mean([balanced_accuracy_value, f1_score_value, roc_auc_score_value, g_mean_score_value, cohen_kappa_value]), 3)
    final_score_values.append(final_score_value)
    
    final_score_std_value = round(np.std([balanced_accuracy_value, f1_score_value, roc_auc_score_value, g_mean_score_value, cohen_kappa_value]), 3)
    final_score_std_values.append(final_score_std_value)
    
    print(f"Fold {i+1}: balanced_accuracy = {round(balanced_accuracy_value, 3)}, f1_score = {round(f1_score_value, 3)}, roc_auc_score = {round(roc_auc_score_value, 3)}, g_mean_score = {round(g_mean_score_value, 3)}, cohen_kappa = {round(cohen_kappa_value, 3)}, final score = {final_score_value}, final score std = {final_score_std_value}")


balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
roc_auc = round(np.mean(scores['test_roc_auc']),3)
g_mean_score = round(np.mean(scores['test_g_mean']),3)
cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)

balanced_accuracy_std = round(np.std(scores['test_balanced_accuracy']),3)
f1_score_std = round(np.std(scores['test_f1']),3)
roc_auc_score_std = round(np.std(scores['test_roc_auc']),3)
g_mean_score_std = round(np.std(scores['test_g_mean']),3)
cohen_kappa_std = round(np.std(scores['test_cohen_kappa']),3)

print("\nscores:")
print("balanced accuracy :", balanced_accuracy)
print("f1 score :", f1)
print("roc auc  :", roc_auc)
print("geometric mean score :", g_mean_score)
print("cohen kappa  :", cohen_kappa)


# balanced_accuracy = round(balanced_accuracy_score(y_test, model_pred),3)
# f1 = round(f1_score(y_test, model_pred),3)
# roc_auc = round(roc_auc_score(y_test, model_pred),3)
# g_mean_score = round(geometric_mean_score(y_test, model_pred),3)
# cohen_kappa = round(cohen_kappa_score(y_test, model_pred),3)

# print("\npredict:")
# print("balanced accuracy :", balanced_accuracy)
# print("f1 score :", f1)
# print("roc auc  :", roc_auc)
# print("geometric mean score :", g_mean_score)
# print("cohen kappa  :", cohen_kappa)



final_score = round(np.mean([balanced_accuracy,f1,roc_auc,g_mean_score,cohen_kappa]),3)
final_score_std = round(np.std([balanced_accuracy,f1,roc_auc,g_mean_score,cohen_kappa]),3)
print("\nfinal score  :", final_score)


# print("\ntime (s)       :", finish_time)
# finish_time_fmt = str(datetime.timedelta(seconds=round(finish_time,0)))
# print("time (HH:mm:ss):", finish_time_fmt)


# # print("class name: ", tpot.__class__.__name__)
classifier = str(tpot._optimized_pipeline)
classifier = classifier.split("(", 1)[0]
# print("\nclassifier: ", classifier)
# print("\nclassifier: ")


output_app = 'TPOT'

print("")
print("APP:", output_app)
print("classifier:", classifier)
print("finish_time:", finish_time)
print("")

r1 = ResultsSHT(dataset_name, classifier, output_app, finish_time, final_score_values, final_score_std_values)

# write_results_sht(r1)





df_tpot = pd.read_csv(sys.path[0] + "/output/" + "results_TPOT.csv", sep=",")

df_tpot2 = df_tpot.loc[df_tpot['dataset'] == dataset_name]

if df_tpot2.empty :

    df_tpot.loc[len(df_tpot.index)] = [
        dataset_name,
        classifier,
        finish_time,
        balanced_accuracy,
        balanced_accuracy_std,
        f1,
        f1_score_std,
        roc_auc,
        roc_auc_score_std,
        g_mean_score,
        g_mean_score_std,
        cohen_kappa,
        cohen_kappa_std,
        final_score,
        final_score_std
    ]

    df_tpot.to_csv(sys.path[0] + "/output/" + "results_TPOT.csv", sep=",", index=False)

    print("\nTPOT Results written, rows added!","\n")

else:
    print("\nTPOT Results not written!","\n")



print('\n\n----------------------------------finish -', datetime.datetime.now(), '--------------------------------------\n\n')
