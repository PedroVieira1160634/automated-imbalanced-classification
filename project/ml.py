import sys
import time
from decimal import Decimal
import pandas as pd
import numpy as np
import openml.datasets
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer, cohen_kappa_score
from imblearn.metrics import geometric_mean_score
#import csv
#import warnings
#warnings.filterwarnings("ignore")


def execute_ml(dataset_location, id_openml):
    
    if dataset_location:
        df, dataset_name = read_file(dataset_location)
    elif id_openml:
        df, dataset_name = read_file_openml(id_openml)
    else:
        return False
    
    # initial validations
    # ...
    
    X, y, characteristics = features_labels(df, dataset_name)
    
    write_characteristics(characteristics)
    
    # array_balancing = ["-"]
    array_balancing = ["-", "RandomUnderSampler", "RandomOverSampler", "SMOTE"]
    # array_balancing = [
    #     "-", 
    #     "ClusterCentroids", "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN", "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler", "TomekLinks",
    #     "RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "SVMSMOTE",
    #     "SMOTEENN", "SMOTETomek"
    # ]
    
    resultsList = []
    
    for balancing in array_balancing:
        X2, y2 = pre_processing(X, y, balancing) 
        resultsList += classify_evaluate(X2, y2, balancing, dataset_name)

    best_result = find_best_result(resultsList)
    
    print("Best classifier is ", best_result.algorithm, " with ", best_result.balancing, "\n")
    
    write_results(best_result)
    
    return True



def read_file(path):
    return pd.read_csv(path), path.split('/')[-1]



def read_file_openml(id):
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe")

    df = pd.DataFrame(X, columns=attribute_names)
    df["class"] = y
    
    dataset_name = dataset.name + " (id:" + str(id) + ")"
    
    return df, dataset_name



def features_labels(df, dataset_name):
    X = df.iloc[: , :-1]
    y = df.iloc[: , -1:]

    df2 = df.iloc[: , :-1]
    n_rows = len(df)
    n_columns = len(X.columns)
    n_numeric_col = X.select_dtypes(include=np.number).shape[1]
    n_categorical_col = X.select_dtypes(include=object).shape[1]

    encoded_columns = []
    for column_name in X.columns:
        if X[column_name].dtype == object or X[column_name].dtype.name == 'category':
            encoded_columns.extend([column_name])
        else:
            pass
    
    if encoded_columns:
        X = pd.get_dummies(X, X[encoded_columns].columns, drop_first=True)

    encoded_columns = []
    preserve_name = ""
    for column_name in y.columns:
        if y[column_name].dtype == object or y[column_name].dtype.name == 'category':
            encoded_columns.extend([column_name])
            preserve_name = column_name
        else:
            pass
    
    if encoded_columns:
        y = pd.get_dummies(y, y[encoded_columns].columns, drop_first=True)

    if preserve_name:
        y.rename(columns={y.columns[0]: preserve_name}, inplace = True)


    imbalance_ratio = 0
    if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
        if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
            imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
        else:
            imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)
    
    df2['Class'] = y
    corr = df2.corr().abs()
    corr = corr.iloc[: , -1].iloc[:-1]

    corr_min, corr_mean, corr_max = 0, 0, 0
    if not corr.empty:
        corr_min = round(corr.min(),3)
        corr_mean = round(corr.mean(),3)
        corr_max = round(corr.max(),3)
    
    df2 = df.iloc[: , :-1]
    list_unique_values = []
    for column in df2:
        if df2[column].dtype == object:
            list_unique_values.append(df2[column].nunique())

    unique_values_min, unique_values_mean, unique_values_max = 0, 0, 0
    if list_unique_values:
        unique_values_min = np.min(list_unique_values)
        unique_values_mean = Decimal(round(np.mean(list_unique_values),0))
        unique_values_max = np.max(list_unique_values)
    
    
    characteristics = Characteristics(dataset_name, n_rows, n_columns, n_numeric_col, n_categorical_col, imbalance_ratio, corr_min, corr_mean, corr_max, unique_values_min, unique_values_mean, unique_values_max)
    
    return X, y, characteristics



def pre_processing(X, y, balancing):
    
    # -- Under-sampling methods --
    if balancing == "ClusterCentroids":
        cc = ClusterCentroids(random_state=42)
        X, y = cc.fit_resample(X, y)

    if balancing == "CondensedNearestNeighbour":
        cnn = CondensedNearestNeighbour(random_state=42, n_jobs=-1) 
        X, y = cnn.fit_resample(X, y)

    if balancing == "EditedNearestNeighbours":
        enn = EditedNearestNeighbours(n_jobs=-1)
        X, y = enn.fit_resample(X, y)

    if balancing == "RepeatedEditedNearestNeighbours":
        renn = RepeatedEditedNearestNeighbours(n_jobs=-1)
        X, y = renn.fit_resample(X, y)

    if balancing == "AllKNN":
        allknn = AllKNN(n_jobs=-1)
        X, y = allknn.fit_resample(X, y)

    if balancing == "InstanceHardnessThreshold":
        iht = InstanceHardnessThreshold(random_state=42, n_jobs=-1)
        X, y = iht.fit_resample(X, y)

    if balancing == "NearMiss":
        nm = NearMiss(n_jobs=-1)
        X, y = nm.fit_resample(X, y)

    if balancing == "NeighbourhoodCleaningRule":
        ncr = NeighbourhoodCleaningRule(n_jobs=-1)
        X, y = ncr.fit_resample(X, y)

    if balancing == "OneSidedSelection":
        oss = OneSidedSelection(random_state=42, n_jobs=-1)
        X, y = oss.fit_resample(X, y)

    if balancing == "RandomUnderSampler":
        rus = RandomUnderSampler(random_state=42) #sampling_strategy=0.5
        X, y = rus.fit_resample(X, y)
    
    if balancing == "TomekLinks":
        tl = TomekLinks(n_jobs=-1)
        X, y = tl.fit_resample(X, y)
    
    
    # -- Over-sampling methods --
    if balancing == "RandomOverSampler":
        over = RandomOverSampler(random_state=42) #sampling_strategy=0.5
        X, y = over.fit_resample(X, y)
    
    if balancing == "SMOTE":
        smote = SMOTE(random_state=42, n_jobs=-1) #sampling_strategy=0.5
        X, y = smote.fit_resample(X, y)
    
    if balancing == "ADASYN":
        ada = ADASYN(random_state=42, n_jobs=-1)
        X, y = ada.fit_resample(X, y)
    
    if balancing == "BorderlineSMOTE":
        sm = BorderlineSMOTE(random_state=42, n_jobs=-1)
        X, y = sm.fit_resample(X, y)
    
    if balancing == "KMeansSMOTE":
        #UserWarning: MiniBatchKMeans
        sm = KMeansSMOTE(random_state=42, n_jobs=-1)
        X, y = sm.fit_resample(X, y)
    
    if balancing == "SVMSMOTE":
        sm = SVMSMOTE(random_state=42, n_jobs=-1)
        X, y = sm.fit_resample(X, y)
    
    
    # -- Combination of over- and under-sampling methods --
    if balancing == "SMOTEENN":
        sme = SMOTEENN(random_state=42, n_jobs=-1)
        X, y = sme.fit_resample(X, y)
        
    if balancing == "SMOTETomek":
        smt = SMOTETomek(random_state=42, n_jobs=-1)
        X, y = smt.fit_resample(X, y)
    
    return X, y



def classify_evaluate(X, y, balancing, dataset_name):

    array_classifiers = [
        #LogisticRegression(random_state=42,max_iter=10000)
        #,GaussianNB() #no random_state (naive bayes)
        #,SVC(random_state=42) 
        #,KNeighborsClassifier() #no random_state
        LGBMClassifier(random_state=42, objective='binary', class_weight='balanced', n_jobs=-1) 
        ,XGBClassifier(random_state=42, use_label_encoder=False, objective='binary:logistic', eval_metric='logloss', n_jobs=-1) #eval_metric=f1_score ; gpu; gpu_predictor
        ,RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        ,ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        ,AdaBoostClassifier(random_state=42)
        ,BaggingClassifier(random_state=42, n_jobs=-1)
        ,GradientBoostingClassifier(random_state=42)
    ]
    
    resultsList = []
    
    #str_balancing = string_balancing(balancing)
    
    for classifier in array_classifiers:
        
        start_time = time.time()
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        
        scoring = {
            'balanced_accuracy': 'balanced_accuracy',
            'f1': 'f1', 
            'roc_auc': 'roc_auc',
            'g_mean': make_scorer(geometric_mean_score, greater_is_better=True),
            'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True)
            }
        
        scores = cross_validate(classifier, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1) #, return_train_score=True
        
        finish_time = round(time.time() - start_time,3)
        
        balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
        f1_score = round(np.mean(scores['test_f1']),3)
        roc_auc_score = round(np.mean(scores['test_roc_auc']),3)
        g_mean = round(np.mean(scores['test_g_mean']),3)
        cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)

        r1 = Results(dataset_name, balancing, classifier.__class__.__name__, finish_time, balanced_accuracy, f1_score, roc_auc_score, g_mean, cohen_kappa)
        resultsList.append(r1)
        
        #print's
        #print("algorithm:", str_balancing + classifier.__class__.__name__)
        # print("Balanced Accuracy    :", balanced_accuracy)
        # print("F1 Score             :", f1_score)
        # print("ROC AUC              :", roc_auc_score)
        # print("G-Mean               :", g_mean)
        # print("Cohen Kappa          :", cohen_kappa)
        #print("time:", finish_time)
        #print("")
        
    
    #print("--")
    
    return resultsList



def find_best_result(resultsList):
    return max(resultsList, key=lambda Results: Results.f1_score)



def string_balancing(balancing):
    str_balancing = ""
    if balancing != "-":
        str_balancing = balancing + " with "
    return str_balancing



def write_characteristics(characteristics):

    print("Write Characteristics")
    
    df_kb_c = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")
    #print(df_kb_c, '\n')

    df_kb_c2 = df_kb_c.loc[df_kb_c['dataset'] == characteristics.dataset_name]
    
    if not df_kb_c2.empty:

        index = df_kb_c2.index.values[0]
        
        if( df_kb_c2.loc[index, 'instances number'] == characteristics.n_rows and 
            df_kb_c2.loc[index, 'attributes number'] == characteristics.n_columns and
            df_kb_c2.loc[index, 'numerical attributes'] == characteristics.n_numeric_col and 
            df_kb_c2.loc[index, 'categorical attributes'] == characteristics.n_categorical_col and
            df_kb_c2.loc[index, 'imbalance ratio'] == characteristics.imbalance_ratio and
            df_kb_c2.loc[index, 'minimum numerical correlation'] == characteristics.corr_min and
            df_kb_c2.loc[index, 'average numerical correlation'] == characteristics.corr_mean and
            df_kb_c2.loc[index, 'maximum numerical correlation'] == characteristics.corr_max and 
            df_kb_c2.loc[index, 'minimum distinct instances in categorical attributes'] == characteristics.unique_values_min and
            df_kb_c2.loc[index, 'average distinct instances in categorical attributes'] == characteristics.unique_values_mean and
            df_kb_c2.loc[index, 'maximum distinct instances in categorical attributes'] == characteristics.unique_values_max
            
        ):
            print("File not written!","\n")
            
        else:
            
            df_kb_c.at[index, 'instances number'] = characteristics.n_rows
            df_kb_c.at[index, 'attributes number'] = characteristics.n_columns
            df_kb_c.at[index, 'numerical attributes'] = characteristics.n_numeric_col
            df_kb_c.at[index, 'categorical attributes'] = characteristics.n_categorical_col
            df_kb_c.at[index, 'imbalance ratio'] = characteristics.imbalance_ratio
            df_kb_c.at[index, 'minimum numerical correlation'] = characteristics.corr_min
            df_kb_c.at[index, 'average numerical correlation'] = characteristics.corr_mean
            df_kb_c.at[index, 'maximum numerical correlation'] = characteristics.corr_max
            df_kb_c.at[index, 'minimum distinct instances in categorical attributes'] = characteristics.unique_values_min
            df_kb_c.at[index, 'average distinct instances in categorical attributes'] = characteristics.unique_values_mean
            df_kb_c.at[index, 'maximum distinct instances in categorical attributes'] = characteristics.unique_values_max

            print("File written, row updated!","\n")
            df_kb_c.to_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",", index=False)
        
    else:

        df_kb_c.loc[len(df_kb_c.index)] = [
            characteristics.dataset_name,
            characteristics.n_rows,
            characteristics.n_columns,
            characteristics.n_numeric_col,
            characteristics.n_categorical_col,
            characteristics.imbalance_ratio,
            characteristics.corr_min,
            characteristics.corr_mean,
            characteristics.corr_max,
            characteristics.unique_values_min,
            characteristics.unique_values_mean,
            characteristics.unique_values_max
        ]

        print("File written, row added!","\n")
        df_kb_c.to_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",", index=False)    



def write_results(best_result):

    print("Best Result Obtained :", float(best_result.f1_score), "\n")
    
    print("Write Results")
    
    df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")

    df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == best_result.dataset_name]
    
    if not df_kb_r2.empty :
        
        if not df_kb_r2[best_result.f1_score > df_kb_r2['f1 score']].empty:
        
            index = df_kb_r2.index.values[0]
            df_kb_r.at[index, 'pre processing'] = best_result.balancing
            df_kb_r.at[index, 'algorithm'] = best_result.algorithm
            df_kb_r.at[index, 'time'] = best_result.time
            df_kb_r.at[index, 'balanced accuracy'] = best_result.balanced_accuracy
            df_kb_r.at[index, 'f1 score'] = best_result.f1_score
            df_kb_r.at[index, 'roc auc'] = best_result.roc_auc_score
            df_kb_r.at[index, 'geometric mean'] = best_result.g_mean_score
            df_kb_r.at[index, 'cohen kappa'] = best_result.cohen_kappa_score
            
            print("File written, row updated!","\n")
            
            df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)
        
        else:
            print("File not written!","\n")
            
    else:

        df_kb_r.loc[len(df_kb_r.index)] = [
            best_result.dataset_name,
            best_result.balancing,
            best_result.algorithm,
            best_result.time,
            best_result.balanced_accuracy, 
            best_result.f1_score, 
            best_result.roc_auc_score,
            best_result.g_mean_score,
            best_result.cohen_kappa_score
        ]

        print("File written, row added!","\n")
        
        df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)    



class Characteristics(object):
    def __init__(self, dataset_name, n_rows, n_columns, n_numeric_col, n_categorical_col, imbalance_ratio, corr_min, corr_mean, corr_max, unique_values_min, unique_values_mean, unique_values_max):
        self.dataset_name = dataset_name
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_numeric_col = n_numeric_col
        self.n_categorical_col = n_categorical_col
        self.imbalance_ratio = imbalance_ratio
        self.corr_min = corr_min
        self.corr_mean = corr_mean
        self.corr_max = corr_max
        self.unique_values_min = unique_values_min
        self.unique_values_mean = unique_values_mean
        self.unique_values_max = unique_values_max
        

class Results(object):
    def __init__(self, dataset_name, balancing, algorithm, time, balanced_accuracy, f1_score, roc_auc_score, g_mean_score, cohen_kappa_score):
        self.dataset_name = dataset_name
        self.balancing = balancing
        self.algorithm = algorithm
        self.time = time
        self.balanced_accuracy = balanced_accuracy
        self.f1_score = f1_score
        self.roc_auc_score = roc_auc_score
        self.g_mean_score = g_mean_score
        self.cohen_kappa_score = cohen_kappa_score
