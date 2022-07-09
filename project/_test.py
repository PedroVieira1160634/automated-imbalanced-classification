#other ML tests

import sys
import time
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder, LabelBinarizer
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

#glass1.dat page-blocks0.dat car-good.dat
dataset_name = "car-good.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)

#1069 1056
# df, dataset_name = read_file_openml(1056)

# print(df)



# X, y, characteristics = features_labels(df, dataset_name)

X = df.iloc[: , :-1]
y = df.iloc[: , -1:]


#TODO ver melhor
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



# #TODO ver parametros
# categorical_ohe_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False) #sparse=False
# # categorical_le_transformer = LabelEncoder() #LabelBinarizer

# # print(df.select_dtypes(include=object)) #["category","object"])

# cat_cols = X.select_dtypes(include=["category","object","bool"]).columns.tolist()
# # cat_cols_X = X.select_dtypes(include=["category","object","bool"]).columns.tolist()
# # cat_cols_y = y.select_dtypes(include=["category","object","bool"]).columns.tolist()

# # print(type(cat_cols))
# # print(cat_cols_X)
# # print(cat_cols_y)

# #TODO ver parametros
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("cat_ohe", categorical_ohe_transformer, cat_cols) #selector(dtype_include=["category","object","bool"])
#         # ,("cat_le", categorical_le_transformer, cat_cols_y)
#     ],
#     remainder='passthrough'
# )

# df_print = pd.DataFrame(preprocessor.fit_transform(df)) #.toarray()
# # pd.set_option('display.max_rows', df_print.shape[0]+1)
# print(df_print)

#balancing_technique = None
balancing_technique = RandomUnderSampler(random_state=42)
classifier = ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

#verbose=True
model = make_pipeline(
    balancing_technique,
    classifier
)


#TODO ver como adiciona ao Pipeline ou make_pipeline

# model = make_pipeline(
#     OneHotEncoder(),
#     RandomOverSampler(random_state=42),
#     ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
# )






# print(characteristics.imbalance_ratio)

#print(y.value_counts())

# over = RandomOverSampler(random_state=42)
# X, y = over.fit_resample(X, y)

#print(y.value_counts())




# print("\nFinal Results:\n")



#   --- k-Fold Cross-Validation ---

start_time = time.time()

#algorithm -> model
# algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
# algorithm = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

scoring = {'balanced_accuracy': 'balanced_accuracy',
           'f1': 'f1',
           'roc_auc': 'roc_auc',
           'g_mean': make_scorer(geometric_mean_score, greater_is_better=True),
           'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True)
           }

#, return_train_score=True return_estimator=True
scores = cross_validate(model, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1, return_train_score=True)

finish_time = round(time.time() - start_time,3)


balanced_accuracy = round(np.mean(scores['train_balanced_accuracy']),3)
f1 = round(np.mean(scores['train_f1']),3)
roc_auc = round(np.mean(scores['train_roc_auc']),3)
g_mean = round(np.mean(scores['train_g_mean']),3)
cohen_kappa = round(np.mean(scores['train_cohen_kappa']),3)

balanced_accuracy_std = round(np.std(scores['train_balanced_accuracy']),3)
f1_std = round(np.std(scores['train_f1']),3)
roc_auc_std = round(np.std(scores['train_roc_auc']),3)
g_mean_std = round(np.std(scores['train_g_mean']),3)
cohen_kappa_std = round(np.std(scores['train_cohen_kappa']),3)

print("train:")
print("Balanced Accuracy    :", balanced_accuracy, " +/- ", balanced_accuracy_std)
print("F1 Score             :", f1, " +/- ", f1_std)
print("ROC AUC              :", roc_auc, " +/- ", roc_auc_std)
print("G-Mean               :", g_mean, " +/- ", g_mean_std)
print("Cohen Kappa          :", cohen_kappa, " +/- ", cohen_kappa_std)
print("")


balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
f1 = round(np.mean(scores['test_f1']),3)
roc_auc = round(np.mean(scores['test_roc_auc']),3)
g_mean = round(np.mean(scores['test_g_mean']),3)
cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)

balanced_accuracy_std = round(np.std(scores['test_balanced_accuracy']),3)
f1_std = round(np.std(scores['test_f1']),3)
roc_auc_std = round(np.std(scores['test_roc_auc']),3)
g_mean_std = round(np.std(scores['test_g_mean']),3)
cohen_kappa_std = round(np.std(scores['test_cohen_kappa']),3)

print("test:")
print("Balanced Accuracy    :", balanced_accuracy, " +/- ", balanced_accuracy_std)
print("F1 Score             :", f1, " +/- ", f1_std)
print("ROC AUC              :", roc_auc, " +/- ", roc_auc_std)
print("G-Mean               :", g_mean, " +/- ", g_mean_std)
print("Cohen Kappa          :", cohen_kappa, " +/- ", cohen_kappa_std)


print("")
print("time                 :", finish_time)

#   --- k-Fold Cross-Validation ---




#   --- train/test ---

# # print(X)
# # print(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# start_time = time.time()

# # print(y_test)

# #comment

# #algorithm -> model
# algorithm = model.fit(X_train, y_train.values.ravel())
# # algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1).fit(X_train, y_train.values.ravel())


# finish_time = round(time.time() - start_time, 3)


# #see if overfitting
# # algorithm_pred = model.predict(X_train)

# # accuracy_score = round(accuracy_score(y_train, algorithm_pred),3)
# # f1_score = round(f1_score(y_train, algorithm_pred),3)
# # roc_auc_score = round(roc_auc_score(y_train, algorithm_pred),3)

# # print("train")
# # print("metric_accuracy:         ", accuracy_score)
# # print("metric_f1_score          ", f1_score)
# # print("metric_roc_auc_score:    ", roc_auc_score)

# algorithm_pred = model.predict(X_test)

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


# print(y_test)
# print(y_test.nunique())

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


# def find_best_result(resultsList):
#     return max(resultsList, key=lambda Results: Results.f1_score)


# class Characteristics(object):
#     def __init__(self, dataset_name, n_rows, n_columns, n_numeric_col, n_categorical_col, imbalance_ratio, corr_min, corr_mean, corr_max, unique_values_min, unique_values_mean, unique_values_max):
#         self.dataset_name = dataset_name
#         self.n_rows = n_rows
#         self.n_columns = n_columns
#         self.n_numeric_col = n_numeric_col
#         self.n_categorical_col = n_categorical_col
#         self.imbalance_ratio = imbalance_ratio
#         self.corr_min = corr_min
#         self.corr_mean = corr_mean
#         self.corr_max = corr_max
#         self.unique_values_min = unique_values_min
#         self.unique_values_mean = unique_values_mean
#         self.unique_values_max = unique_values_max


# def features_labels(df, dataset_name):
    
#     print("Dataset                      :", dataset_name, "\n")
    
#     X = df.iloc[: , :-1]
#     y = df.iloc[: , -1:]

#     df2 = df.iloc[: , :-1]
#     n_rows = len(df)
#     n_columns = len(X.columns)
#     n_numeric_col = X.select_dtypes(include=np.number).shape[1]
#     n_categorical_col = X.select_dtypes(include=object).shape[1]

#     encoded_columns = []
#     for column_name in X.columns:
#         if X[column_name].dtype == object or X[column_name].dtype.name == 'category':
#             encoded_columns.extend([column_name])
    
#     if encoded_columns:
#         X = pd.get_dummies(X, X[encoded_columns].columns, drop_first=True)

#     encoded_columns = []
#     preserve_name = ""
#     for column_name in y.columns:
#         if y[column_name].dtype == object or y[column_name].dtype.name == 'category':
#             encoded_columns.extend([column_name])
#             preserve_name = column_name
    
#     if encoded_columns:
#         y = pd.get_dummies(y, y[encoded_columns].columns, drop_first=True)

#     if preserve_name:
#         y.rename(columns={y.columns[0]: preserve_name}, inplace = True)


#     imbalance_ratio = 0
#     if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
#         if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
#             imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
#         else:
#             imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)
    
#     df2['Class'] = y
#     corr = df2.corr().abs()
#     corr = corr.iloc[: , -1].iloc[:-1]

#     corr_min, corr_mean, corr_max = 0, 0, 0
#     if not corr.empty:
#         corr_min = round(corr.min(),3)
#         corr_mean = round(corr.mean(),3)
#         corr_max = round(corr.max(),3)
    
#     df2 = df.iloc[: , :-1]
#     list_unique_values = []
#     for column in df2:
#         if df2[column].dtype == object:
#             list_unique_values.append(df2[column].nunique())

#     unique_values_min, unique_values_mean, unique_values_max = 0, 0, 0
#     if list_unique_values:
#         unique_values_min = np.min(list_unique_values)
#         unique_values_mean = Decimal(round(np.mean(list_unique_values),0))
#         unique_values_max = np.max(list_unique_values)
    
    
#     characteristics = Characteristics(dataset_name, n_rows, n_columns, n_numeric_col, n_categorical_col, imbalance_ratio, corr_min, corr_mean, corr_max, unique_values_min, unique_values_mean, unique_values_max)
    
#     return X, y, characteristics


# def pre_processing(X, y, balancing):
    
#     # -- Under-sampling methods --
#     if balancing == "ClusterCentroids":
#         cc = ClusterCentroids(random_state=42)
#         X, y = cc.fit_resample(X, y)

#     if balancing == "CondensedNearestNeighbour":
#         cnn = CondensedNearestNeighbour(random_state=42, n_jobs=-1) 
#         X, y = cnn.fit_resample(X, y)

#     if balancing == "EditedNearestNeighbours":
#         enn = EditedNearestNeighbours(n_jobs=-1)
#         X, y = enn.fit_resample(X, y)

#     if balancing == "RepeatedEditedNearestNeighbours":
#         renn = RepeatedEditedNearestNeighbours(n_jobs=-1)
#         X, y = renn.fit_resample(X, y)

#     if balancing == "AllKNN":
#         allknn = AllKNN(n_jobs=-1)
#         X, y = allknn.fit_resample(X, y)

#     if balancing == "InstanceHardnessThreshold":
#         iht = InstanceHardnessThreshold(random_state=42, n_jobs=-1)
#         X, y = iht.fit_resample(X, y)

#     if balancing == "NearMiss":
#         nm = NearMiss(n_jobs=-1)
#         X, y = nm.fit_resample(X, y)

#     if balancing == "NeighbourhoodCleaningRule":
#         ncr = NeighbourhoodCleaningRule(n_jobs=-1)
#         X, y = ncr.fit_resample(X, y)

#     if balancing == "OneSidedSelection":
#         oss = OneSidedSelection(random_state=42, n_jobs=-1)
#         X, y = oss.fit_resample(X, y)

#     if balancing == "RandomUnderSampler":
#         rus = RandomUnderSampler(random_state=42) #sampling_strategy=0.5
#         X, y = rus.fit_resample(X, y)
    
#     if balancing == "TomekLinks":
#         tl = TomekLinks(n_jobs=-1)
#         X, y = tl.fit_resample(X, y)
    
    
#     # -- Over-sampling methods --
#     if balancing == "RandomOverSampler":
#         over = RandomOverSampler(random_state=42) #sampling_strategy=0.5
#         X, y = over.fit_resample(X, y)
    
#     if balancing == "SMOTE":
#         smote = SMOTE(random_state=42, n_jobs=-1) #sampling_strategy=0.5
#         X, y = smote.fit_resample(X, y)
    
#     if balancing == "ADASYN":
#         ada = ADASYN(random_state=42, n_jobs=-1)
#         X, y = ada.fit_resample(X, y)
    
#     if balancing == "BorderlineSMOTE":
#         sm = BorderlineSMOTE(random_state=42, n_jobs=-1)
#         X, y = sm.fit_resample(X, y)
    
#     if balancing == "KMeansSMOTE":
#         #UserWarning: MiniBatchKMeans
#         # kmeans = MiniBatchKMeans(batch_size=2048)
#         # , kmeans_estimator=kmeans
        
#         imbalance_ratio = 0
#         if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
#             if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
#                 imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
#             else:
#                 imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)
        
#         n_clusters = 1/imbalance_ratio
        
#         sm = KMeansSMOTE(random_state=42, n_jobs=-1, cluster_balance_threshold=n_clusters)
#         X, y = sm.fit_resample(X, y)
    
#     if balancing == "SVMSMOTE":
#         sm = SVMSMOTE(random_state=42, n_jobs=-1)
#         X, y = sm.fit_resample(X, y)
    
    
#     # -- Combination of over- and under-sampling methods --
#     if balancing == "SMOTEENN":
#         sme = SMOTEENN(random_state=42, n_jobs=-1)
#         X, y = sme.fit_resample(X, y)
        
#     if balancing == "SMOTETomek":
#         smt = SMOTETomek(random_state=42, n_jobs=-1)
#         X, y = smt.fit_resample(X, y)
    
#     return X, y


# def write_characteristics(characteristics, best_result):

#     if not characteristics or not best_result:
#         print("--characteristics or best_result not valid on write_characteristics--")
#         print("characteristics:", characteristics)
#         print("best_result:", best_result)
#         return False
    
#     try:
        
#         df_kb_c = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")
#         #print(df_kb_c, '\n')

#         df_kb_c2 = df_kb_c.loc[df_kb_c['dataset'] == characteristics.dataset_name]
        
#         if not df_kb_c2.empty:
            
#             index = df_kb_c2.index.values[0]
            
#             if( df_kb_c2.loc[index, 'instances number'] == characteristics.n_rows and 
#                 df_kb_c2.loc[index, 'attributes number'] == characteristics.n_columns and
#                 df_kb_c2.loc[index, 'numerical attributes'] == characteristics.n_numeric_col and 
#                 df_kb_c2.loc[index, 'categorical attributes'] == characteristics.n_categorical_col and
#                 df_kb_c2.loc[index, 'imbalance ratio'] == characteristics.imbalance_ratio and
#                 df_kb_c2.loc[index, 'minimum numerical correlation'] == characteristics.corr_min and
#                 df_kb_c2.loc[index, 'average numerical correlation'] == characteristics.corr_mean and
#                 df_kb_c2.loc[index, 'maximum numerical correlation'] == characteristics.corr_max and 
#                 df_kb_c2.loc[index, 'minimum distinct instances in categorical attributes'] == characteristics.unique_values_min and
#                 df_kb_c2.loc[index, 'average distinct instances in categorical attributes'] == characteristics.unique_values_mean and
#                 df_kb_c2.loc[index, 'maximum distinct instances in categorical attributes'] == characteristics.unique_values_max and
#                 df_kb_c2.loc[index, 'pre processing'] == best_result.balancing and
#                 df_kb_c2.loc[index, 'algorithm'] == best_result.algorithm
#             ):
#                 print("Write Characteristics not written!","\n")
                
#             else:
                
#                 df_kb_c.at[index, 'instances number'] = characteristics.n_rows
#                 df_kb_c.at[index, 'attributes number'] = characteristics.n_columns
#                 df_kb_c.at[index, 'numerical attributes'] = characteristics.n_numeric_col
#                 df_kb_c.at[index, 'categorical attributes'] = characteristics.n_categorical_col
#                 df_kb_c.at[index, 'imbalance ratio'] = characteristics.imbalance_ratio
#                 df_kb_c.at[index, 'minimum numerical correlation'] = characteristics.corr_min
#                 df_kb_c.at[index, 'average numerical correlation'] = characteristics.corr_mean
#                 df_kb_c.at[index, 'maximum numerical correlation'] = characteristics.corr_max
#                 df_kb_c.at[index, 'minimum distinct instances in categorical attributes'] = characteristics.unique_values_min
#                 df_kb_c.at[index, 'average distinct instances in categorical attributes'] = characteristics.unique_values_mean
#                 df_kb_c.at[index, 'maximum distinct instances in categorical attributes'] = characteristics.unique_values_max
#                 df_kb_c.at[index, 'pre processing'] = best_result.balancing
#                 df_kb_c.at[index, 'algorithm'] = best_result.algorithm
                
#                 df_kb_c.to_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",", index=False)
                
#                 print("Write Characteristics written, row updated!","\n")
            
#         else:
            
#             df_kb_c.loc[len(df_kb_c.index)] = [
#                 characteristics.dataset_name,
#                 characteristics.n_rows,
#                 characteristics.n_columns,
#                 characteristics.n_numeric_col,
#                 characteristics.n_categorical_col,
#                 characteristics.imbalance_ratio,
#                 characteristics.corr_min,
#                 characteristics.corr_mean,
#                 characteristics.corr_max,
#                 characteristics.unique_values_min,
#                 characteristics.unique_values_mean,
#                 characteristics.unique_values_max,
#                 best_result.balancing,
#                 best_result.algorithm
#             ]

#             df_kb_c.to_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",", index=False)
            
#             print("Write Characteristics written, row added!","\n")

#         return True
    
#     except Exception as e:
#         print("--Did NOT Wrote characteristics on write_characteristics--")
#         print(e)
#         return False


# def write_results_elapsed_time(elapsed_time, dataset_name):

#     if not elapsed_time or not dataset_name:
#         print("--elapsed_time or dataset_name not valid on write_results_elapsed_time--")
#         print("elapsed_time:", elapsed_time)
#         print("dataset_name:", dataset_name)
#         return False

#     try:

#         print("Write Result (elapsed time)")

#         df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")

#         df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == dataset_name]

#         if not df_kb_r2.empty :
#             index = df_kb_r2.index.values[0]
#             elapsed_time = str(datetime.timedelta(seconds=round(elapsed_time,0)))
#             df_kb_r.at[index, 'total elapsed time'] = elapsed_time

#             df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)

#             print("File written, row updated!","\n")

#         else:
#             print("File not written!","\n")

#         return True

#     except Exception as e:
#         print("--Did NOT Wrote elapsed_time on write_results_elapsed_time--")
#         print(e)
#         return False


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

