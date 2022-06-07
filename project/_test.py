#example ML

import sys
import time
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer, cohen_kappa_score, average_precision_score
from imblearn.metrics import geometric_mean_score

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')


#glass1.dat
#page-blocks0.dat
#car-good.dat
#kddcup-rootkit-imap_vs_back.dat #to be removed, only for tests
dataset_name = "glass1.dat"
df = pd.read_csv(sys.path[0] + "/input/" + dataset_name)

X = df.iloc[: , :-1]
y = df.iloc[: , -1:]


# -- characteristics of datasets

df2 = df.iloc[: , :-1]

n_rows = len(df)
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

print("number of rows       :", n_rows)
print("number of columns    :", n_columns)
print("")

print("numeric columns      :", n_numeric_col)
print("non-numeric columns  :", n_non_numeric_col)
print("")




# correlation ignoring categorical columns, and only with label (last) column

df2['Class'] = y

corr = df2.corr().abs()
corr = corr.iloc[: , -1].iloc[:-1]

corr_max, corr_mean, corr_min = 0, 0, 0
if not corr.empty:
    corr_max = round(corr.max(),3)
    corr_mean = round(corr.mean(),3)
    corr_min = round(corr.min(),3)
    
print("maximum correlation  :", corr_max)
print("average correlation  :", corr_mean)
print("minimum correlation  :", corr_min)
print("")


#distinct with only categorical columns, ignoring the label (last) column

print("distinct values in columns:")
df2 = df.iloc[: , :-1]
# for column in df2:
#     if df2[column].dtype == object:
#         print(column, " - ", df2[column].nunique())
# print("")

list_unique_values = []
for column in df2:
    if df2[column].dtype == object:
        list_unique_values.append(df2[column].nunique())

mean_unique_values = 0
if list_unique_values:
    mean_unique_values = Decimal(round(np.mean(list_unique_values),0))

print("average of distinct values in columns:", mean_unique_values)
print("")


imbalance_ratio = 0
if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
    if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
        imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
    else:
        imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)

print("imbalance ratio      :", imbalance_ratio)
print("")



#normalize

# scaler = preprocessing.MinMaxScaler()
# names = X.columns
# d = scaler.fit_transform(X)
# scaled_X = pd.DataFrame(d, columns=names)
# X = scaled_X




#print(y.value_counts())

smote = SMOTE(random_state=42) #, k_neighbors=minimum_samples
X, y = smote.fit_resample(X, y)

#print(y.value_counts())



# print("\nFinal Results:\n")

#   --- k-Fold Cross-Validation ---

start_time = time.time()

algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced') #.fit(X_train, y_train.values.ravel())
#algorithm = KNeighborsClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

scoring = {'balanced_accuracy': 'balanced_accuracy',
           'f1': 'f1', 
           'roc_auc': 'roc_auc',
           'g_mean': make_scorer(geometric_mean_score, greater_is_better=True),
           'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True)
           }

#, return_train_score=True
scores = cross_validate(algorithm, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1)

finish_time = round(time.time() - start_time,3)

#print("Mean F1 Score        : %.3f" % np.mean(scores_f1))
#print("Mean ROC AUC Score   : %.3f" % np.mean(scores_roc_auc))

# print("train:")
# print("Mean Accuracy Score  : %.3f" % np.mean(scores['train_balanced_accuracy']))
# print("Mean F1 Score        : %.3f" % np.mean(scores['train_f1']))
# print("Mean ROC AUC         : %.3f" % np.mean(scores['train_roc_auc']))
# print("")

balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
roc_auc = round(np.mean(scores['test_roc_auc']),3)
g_mean = round(np.mean(scores['test_g_mean']),3)
cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)

# print("test:")
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

# algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced').fit(X_train, y_train.values.ravel())

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




# print("write kb_characteristics:\n")

# def write_characteristics(dataset_name):

#     df_kb_c = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")
#     #print(df_kb_c, '\n')

#     df_kb_c2 = df_kb_c.loc[df_kb_c['dataset'] == dataset_name]
    
#     if not df_kb_c2.empty:

#         index = df_kb_c2.index.values[0]
#         df_kb_c.at[index, 'rows number'] = n_rows
#         df_kb_c.at[index, 'columns number'] = n_columns
#         df_kb_c.at[index, 'numeric columns'] = n_numeric_col
#         df_kb_c.at[index, 'non-numeric columns'] = n_non_numeric_col
#         df_kb_c.at[index, 'maximum correlation'] = corr_max
#         df_kb_c.at[index, 'average correlation'] = corr_mean
#         df_kb_c.at[index, 'minimum correlation'] = corr_min
#         df_kb_c.at[index, 'average of distinct values in columns'] = mean_unique_values
#         df_kb_c.at[index, 'imbalance ratio'] = imbalance_ratio

#         print("File written, row updated!","\n")

#     else:

#         df_kb_c.loc[len(df_kb_c.index)] = [
#             dataset_name,
#             n_rows,
#             n_columns,
#             n_numeric_col,
#             n_non_numeric_col,
#             corr_max,
#             corr_mean,
#             corr_min,
#             mean_unique_values,
#             imbalance_ratio
#         ]

#         print("File written, row added!","\n")

#     #print(df_kb_c, '\n')
#     df_kb_c.to_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",", index=False)


# write_characteristics(dataset_name)



# print("write kb_results:\n")

# def write_results(dataset_name):

#     balancing = 'SMOTE'
#     algorithm_name = algorithm.__class__.__name__

#     df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")
#     #print(df_kb_r, '\n')

#     df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == dataset_name]
    
#     if not df_kb_r2.empty :
        
#         if not df_kb_r2[f1_score > df_kb_r2['f1 score']].empty:
        
#             index = df_kb_r2.index.values[0]
#             df_kb_r.at[index, 'pre processing'] = balancing
#             df_kb_r.at[index, 'algorithm'] = algorithm_name
#             df_kb_r.at[index, 'time'] = finish_time
#             df_kb_r.at[index, 'accuracy'] = accuracy_score
#             df_kb_r.at[index, 'f1 score'] = f1_score
#             df_kb_r.at[index, 'roc auc'] = roc_auc_score
            
#             print("File written, row updated!","\n")
            
#             #print(df_kb_r, '\n')
#             df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)
        
#         else:
#             print("File not written!","\n")
            
#     else:

#         df_kb_r.loc[len(df_kb_r.index)] = [
#             dataset_name,
#             balancing,
#             algorithm_name,
#             finish_time,
#             accuracy_score, 
#             f1_score, 
#             roc_auc_score
#         ]

#         print("File written, row added!","\n")
        
#         #print(df_kb_r, '\n')
#         df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)
    


# write_results(dataset_name)



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

