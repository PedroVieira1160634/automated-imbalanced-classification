#other tests ML

import sys
import time
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer, cohen_kappa_score, average_precision_score
from imblearn.metrics import geometric_mean_score
from pymfe.mfe import MFE
from ml import read_file, read_file_openml, features_labels
# import warnings
# warnings.filterwarnings("ignore")

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')


dataset_name = "glass1.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

# openml
# df, dataset_name = read_file_openml(450)



#normalize

# scaler = preprocessing.MinMaxScaler()
# names = X.columns
# d = scaler.fit_transform(X)
# scaled_X = pd.DataFrame(d, columns=names)
# X = scaled_X

X, y, characteristics = features_labels(df, dataset_name)



#print(y.value_counts())

# print(characteristics.imbalance_ratio)

n_clusters = 1/characteristics.imbalance_ratio

sm = KMeansSMOTE(random_state=42, n_jobs=-1, cluster_balance_threshold=n_clusters) #0.04 0.0001 k_neighbors=2
X, y = sm.fit_resample(X, y)

# sm = KMeansSMOTE(random_state=42, n_jobs=-1)
# X, y = sm.fit_resample(X, y)

# smote = SMOTE(random_state=42, n_jobs=-1)
# X, y = smote.fit_resample(X, y)

#print(y.value_counts())




# print("\nFinal Results:\n")

#   --- k-Fold Cross-Validation ---

start_time = time.time()

algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

scoring = {'balanced_accuracy': 'balanced_accuracy',
           'f1': 'f1', 
           'roc_auc': 'roc_auc',
           'g_mean': make_scorer(geometric_mean_score, greater_is_better=True),
           'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True)
           }

#, return_train_score=True
scores = cross_validate(algorithm, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1, return_train_score=True)

finish_time = round(time.time() - start_time,3)


balanced_accuracy = round(np.mean(scores['train_balanced_accuracy']),3)
f1 = round(np.mean(scores['train_f1']),3)
roc_auc = round(np.mean(scores['train_roc_auc']),3)
g_mean = round(np.mean(scores['train_g_mean']),3)
cohen_kappa = round(np.mean(scores['train_cohen_kappa']),3)

print("train:")
print("Balanced Accuracy    :", balanced_accuracy)
print("F1 Score             :", f1)
print("ROC AUC              :", roc_auc)
print("G-Mean               :", g_mean)
print("Cohen Kappa          :", cohen_kappa)
print("")

balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
roc_auc = round(np.mean(scores['test_roc_auc']),3)
g_mean = round(np.mean(scores['test_g_mean']),3)
cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)

print("test:")
print("Balanced Accuracy    :", balanced_accuracy)
print("F1 Score             :", f1)
print("ROC AUC              :", roc_auc)
print("G-Mean               :", g_mean)
print("Cohen Kappa          :", cohen_kappa)

print("")
print("time                 :", finish_time)

#   --- k-Fold Cross-Validation ---




#   --- train/test ---

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #0.452 sec (23x more fast)

# start_time = time.time()

# algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1).fit(X_train, y_train.values.ravel())

# finish_time = round(time.time() - start_time, 3)

# algorithm_pred = algorithm.predict(X_train)

# #see if overfitting
# # accuracy_score = round(accuracy_score(y_train, algorithm_pred),3)
# # f1_score = round(f1_score(y_train, algorithm_pred),3)
# # roc_auc_score = round(roc_auc_score(y_train, algorithm_pred),3)

# # print("train")
# # print("metric_accuracy:         ", accuracy_score)
# # print("metric_f1_score          ", f1_score)
# # print("metric_roc_auc_score:    ", roc_auc_score)

# algorithm_pred = algorithm.predict(X_test)

# accuracy_score = round(accuracy_score(y_test, algorithm_pred),3)
# f1_score = round(f1_score(y_test, algorithm_pred),3)
# roc_auc_score = round(roc_auc_score(y_test, algorithm_pred),3)

# #print("test")
# print("metric_accuracy      :", accuracy_score)
# print("metric_f1_score      :", f1_score)
# print("metric_roc_auc_score :", roc_auc_score)
# print("")

# print("time                 :", finish_time)
# print("")

#   --- train/test ---



print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')



# old code


#correlation all columns
# corr = df.corr().abs()
# corr = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#        .stack()
#        .sort_values(ascending=False))

# corr_max = corr.max()
# corr_mean = corr.mean()
# corr_min = corr.min()


#SMOTE
# minimum_samples = min(y.value_counts())
# if minimum_samples >= 5:
#     minimum_samples = 5
# else:
#     minimum_samples -= 1
# smote = SMOTE(random_state=42, k_neighbors=minimum_samples) #sampling_strategy=0.5
# X, y = smote.fit_resample(X, y)


# def write(best_result, dataset_name):
    
#     previous_result, previous_found, found_index = read(dataset_name)
    
#     print("")
#     print("best_result     :", float(best_result.f1_score))
#     print("previous_result :", float(previous_result))
    
#     if previous_found and float(best_result.f1_score) <= float(previous_result):
#         print("File NOT written!","\n")
#         return False
    
#     print("File written!","\n")
    
#     str_balancing = string_balancing(best_result.balancing)
    
#     if not previous_found:
#         with open(sys.path[0] + '/output/results.csv', 'a', newline='') as f:
#            writer = csv.writer(f)
        
#            writer.writerow([best_result.dataset_name, str_balancing + best_result.algorithm])
#            writer.writerow(["accuracy_score", str(best_result.accuracy)])
#            writer.writerow(["f1_score", str(best_result.f1_score)])
#            writer.writerow(["roc_auc_score", str(best_result.roc_auc_score)])
#            writer.writerow(["time", str(best_result.time)])
#            writer.writerow(["---"])
           
#     else:
#         new_file_content = ""
#         with open(sys.path[0] + '/output/results.csv', 'r') as reading_file:
        
#             i = 0
#             j = 5               #4 metrics + 1 seperator
            
#             for line in reading_file:
#                 stripped_line = line.strip()
                
#                 if found_index <= i <= (found_index + j):
#                     if i == found_index:
#                         new_line = stripped_line.replace(stripped_line, best_result.dataset_name + "," + str_balancing + best_result.algorithm)
#                     if i == found_index + 1:
#                         new_line = stripped_line.replace(stripped_line, "accuracy_score" + "," + str(best_result.accuracy))
#                     if i == found_index + 2:
#                         new_line = stripped_line.replace(stripped_line, "f1_score" + "," + str(best_result.f1_score))
#                     if i == found_index + 3:
#                         new_line = stripped_line.replace(stripped_line, "roc_auc_score" + "," + str(best_result.roc_auc_score))
#                     if i == found_index + 4:
#                         new_line = stripped_line.replace(stripped_line, "time" + "," + str(best_result.time))
#                     if i == found_index + 5:
#                         new_line = stripped_line.replace(stripped_line, "---")
#                 else:
#                     new_line = stripped_line
                    
#                 new_file_content += new_line +"\n"
#                 i+=1

#         if new_file_content:
#             with open(sys.path[0] + '/output/results.csv', "w") as writing_file:
#                 writing_file.write(new_file_content)

#     return True
    

# def read(dataset_name):
#     dataset_name = dataset_name + ","
#     selected_metric = "f1_score" + ","
#     result = 0
    
#     with open(sys.path[0] + '/output/results.csv', 'r') as reading_file:

#         i = 0
#         j = 5             #4 metrics + 1 seperator
#         found = False
#         only_once = True
#         found_index = -1
#         found_result = False
        
#         for line in reading_file:
#             if i % (j+1) == 0:
#                 if line.startswith(dataset_name) and only_once == True:
#                     found = True
#                     only_once = False
#                     found_index = i
#                     found_result = True
#                 else:
#                     found = False
#             elif found and line.startswith(selected_metric):
#                 result = line.partition(selected_metric)[2]
#             i+=1
        
#         if found_index == -1:
#             found_index = i
    
#     return result, found_result, found_index

