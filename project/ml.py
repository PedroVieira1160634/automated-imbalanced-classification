import time
import sys
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import csv
#import warnings
#warnings.filterwarnings("ignore")


def execute_ml(dataset_location):
    
    dataset_name = dataset_location.split('/')[-1]
    
    df = read_file(dataset_location)
    
    #array_balancing = ["-"]
    array_balancing = ["-","SMOTE","OVER","UNDER"]
    
    resultsList = []
    
    for balancing in array_balancing:
        X, y = pre_processing(df, balancing) 
        resultsList += classify_evaluate(X, y, dataset_name, balancing)

    best_result = find_best_result(resultsList)
    
    print("Best classifier is ", best_result.balancing, " _ ", best_result.algorithm, "\n")
    
    #write(best_result, dataset_name)
    

def read_file(path):
    return pd.read_csv(path)


def pre_processing(df, balancing):
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
    
    if balancing == "SMOTE":
        minimum_samples = min(y.value_counts())
        if minimum_samples >= 5:
            minimum_samples = 5
        else:
            minimum_samples -= 1
        smote = SMOTE(random_state=42, k_neighbors=minimum_samples) #sampling_strategy=0.5
        X, y = smote.fit_resample(X, y)
    
    if balancing == "OVER":
        over = RandomOverSampler(random_state=42) #sampling_strategy=0.5
        X, y = over.fit_resample(X, y)
     
    if balancing == "UNDER":
        under = RandomUnderSampler(random_state=42) #sampling_strategy=0.5
        X, y = under.fit_resample(X, y)
    
    return X, y


def classify_evaluate(X, y, dataset_name, balancing):

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
        # ,EasyEnsembleClassifier(random_state=42, n_jobs=-1)
        # ,RUSBoostClassifier(random_state=42)
        # ,BalancedBaggingClassifier(random_state=42, n_jobs=-1)
        # ,BalancedRandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        ]
    
    resultsList = []
    
    str_balancing = string_balancing(balancing)
    
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
        str_balancing = balancing + " _ "
    return str_balancing

class Characteristics(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.n_rows = n_rows,
        self.n_columns = n_columns,
        self.n_numeric_col = n_numeric_col,
        self.n_non_numeric_col = n_non_numeric_col,
        self.corr_max = corr_max,
        self.corr_mean = corr_mean,
        self.corr_min = corr_min,
        self.mean_unique_values = mean_unique_values,
        self.imbalance_ratio = imbalance_ratio

class Results(object):
    def __init__(self, dataset_name, balancing, algorithm, time, accuracy, f1_score, roc_auc_score):
        self.dataset_name = dataset_name
        self.balancing = balancing
        self.algorithm = algorithm
        self.time = time
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.roc_auc_score = roc_auc_score
