import sys
import time
from decimal import Decimal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier
#from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
#import csv
#import warnings
#warnings.filterwarnings("ignore")


def execute_ml(dataset_location):
    
    df, dataset_name = read_file(dataset_location)
    
    X, y, characteristics = features_labels(df, dataset_name)
    
    write_characteristics(characteristics)
    
    array_balancing = ["-","RandomUnderSampler","RandomOverSampler","SMOTE"] #["-"]
    resultsList = []
    
    for balancing in array_balancing:
        X2, y2 = pre_processing(X, y, balancing) 
        resultsList += classify_evaluate(X2, y2, balancing, dataset_name)

    best_result = find_best_result(resultsList)
    
    print("Best classifier is ", best_result.algorithm, " with ", best_result.balancing, "\n")
    
    write_results(best_result)
    


def read_file(path):
    return pd.read_csv(path), path.split('/')[-1]



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
        if X[column_name].dtype == object:
            encoded_columns.extend([column_name])
        else:
            pass
    
    X = pd.get_dummies(X, X[encoded_columns].columns, drop_first=True)

    encoded_columns = []
    preserve_name = ""
    for column_name in y.columns:
        if y[column_name].dtype == object:
            encoded_columns.extend([column_name])
            preserve_name = column_name
        else:
            pass
    
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
    
    if balancing == "RandomUnderSampler":
        under = RandomUnderSampler(random_state=42) #sampling_strategy=0.5
        X, y = under.fit_resample(X, y)
    
    if balancing == "RandomOverSampler":
        over = RandomOverSampler(random_state=42) #sampling_strategy=0.5
        X, y = over.fit_resample(X, y)
    
    if balancing == "SMOTE":
        smote = SMOTE(random_state=42) #sampling_strategy=0.5
        X, y = smote.fit_resample(X, y)
    
    return X, y



def classify_evaluate(X, y, balancing, dataset_name):

    array_classifiers = [
        #LogisticRegression(random_state=42,max_iter=10000)
        #,GaussianNB() #(naive bayes) random_state?
        #,SVC(random_state=42) 
        #,KNeighborsClassifier() #random_state?
        LGBMClassifier(random_state=42, objective='binary', class_weight='balanced') 
        ,XGBClassifier(random_state=42, use_label_encoder=False, objective='binary:logistic', eval_metric='logloss') #eval_metric=f1_score ; gpu 
        ,RandomForestClassifier(random_state=42, class_weight='balanced')
        ,ExtraTreesClassifier(random_state=42, class_weight='balanced')
        # ,AdaBoostClassifier(random_state=42)
        # ,BaggingClassifier(random_state=42)
        # ,GradientBoostingClassifier(random_state=42)
        ]
    
    #if balancing = "-"
    # ,EasyEnsembleClassifier(random_state=42, n_jobs=-1)
    # ,RUSBoostClassifier(random_state=42)
    # ,BalancedBaggingClassifier(random_state=42, n_jobs=-1)
    # ,BalancedRandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    resultsList = []
    
    #str_balancing = string_balancing(balancing)
    
    for classifier in array_classifiers:
        
        #start_time = time.time()
        
        #algorithm = classifier.fit(X_train, y_train.values.ravel()) #eval_metric 
        
        #finish_time = (round(time.time() - start_time,3))

        #algorithm_pred = algorithm.predict(X_test)

        # metric_accuracy = round(accuracy_score(y_test, algorithm_pred),3)
        # metric_f1_score = round(f1_score(y_test, algorithm_pred),3)
        # metric_roc_auc_score = round(roc_auc_score(y_test, algorithm_pred),3)
        
        start_time = time.time()
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        
        scoring = {
            'balanced_accuracy': 'balanced_accuracy',
            'f1': 'f1', 
            'roc_auc': 'roc_auc'}
        
        scores = cross_validate(classifier, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1) #, return_train_score=True
        
        finish_time = round(time.time() - start_time,3)
        
        metric_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
        metric_f1_score = round(np.mean(scores['test_f1']),3)
        metric_roc_auc_score = round(np.mean(scores['test_roc_auc']),3)
        
        r1 = Results(dataset_name, balancing, classifier.__class__.__name__, finish_time, metric_accuracy, metric_f1_score, metric_roc_auc_score)
        resultsList.append(r1)
        
        #print's
        #print("algorithm:", str_balancing + classifier.__class__.__name__)
        #print("accuracy_score:", metric_accuracy)
        #print("f1_score:", metric_f1_score)
        #print("roc_auc_score:", metric_roc_auc_score)
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
            df_kb_r.at[index, 'accuracy'] = best_result.accuracy
            df_kb_r.at[index, 'f1 score'] = best_result.f1_score
            df_kb_r.at[index, 'roc auc'] = best_result.roc_auc_score
            
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
            best_result.accuracy, 
            best_result.f1_score, 
            best_result.roc_auc_score
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
    def __init__(self, dataset_name, balancing, algorithm, time, accuracy, f1_score, roc_auc_score):
        self.dataset_name = dataset_name
        self.balancing = balancing
        self.algorithm = algorithm
        self.time = time
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.roc_auc_score = roc_auc_score
