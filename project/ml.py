import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split #, RepeatedStratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
#import lazypredict from lazypredict.Supervised import LazyClassifier
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
        x_train, x_test, y_train, y_test = train_test_split_func(df, balancing)
        resultsList += classify_evaluate(x_train, x_test, y_train, y_test, dataset_name, balancing)

    best_result = find_best_result(resultsList)
    
    print("Best classifier is ", best_result.balancing, " _ ", best_result.algorithm, "\n")
    
    write(best_result, dataset_name)
    

def read_file(path):
    return pd.read_csv(path)


def train_test_split_func(df, balancing):
    y = df.iloc[:,-1:]
    X = df.iloc[:,:-1]

    if balancing == "SMOTE":
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
    
    if balancing == "OVER":
        over = SMOTE(sampling_strategy=0.2)
        X, y = over.fit_resample(X, y)
     
    if balancing == "UNDER":
        under = RandomUnderSampler(sampling_strategy=0.5)
        X, y = under.fit_resample(X, y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def classify_evaluate(x_train, x_test, y_train, y_test, dataset_name, balancing):

    array_classifiers = [
        #LogisticRegression(max_iter=10000)
        #,GaussianNB() #(naive bayes)
        #,svm.SVC() 
        #,KNeighborsClassifier()
        lgb.LGBMClassifier() #objective='binary' 
        ,XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        ,RandomForestClassifier()
        ,ExtraTreesClassifier()
        ]
    
    resultsList = []
    
    str_balancing = string_balancing(balancing)
    
    for classifier in array_classifiers:
        
        start_time = time.time()
        
        algorithm = classifier.fit(x_train, y_train.values.ravel())

        finish_time = (round(time.time() - start_time,5))

        algorithm_pred = algorithm.predict(x_test)

        metric_accuracy = round(accuracy_score(y_test, algorithm_pred),5)
        metric_f1_score = round(f1_score(y_test, algorithm_pred),5)
        metric_roc_auc_score = round(roc_auc_score(y_test, algorithm_pred),5)
        
        r1 = Results(dataset_name, balancing, classifier.__class__.__name__, finish_time, metric_accuracy, metric_f1_score, metric_roc_auc_score)
        resultsList.append(r1)
        
        #print's
        #print("algorithm:", str_balancing + classifier.__class__.__name__)
        #print("accuracy_score:", metric_accuracy)
        #print("f1_score:", metric_f1_score)
        #print("roc_auc_score:", metric_roc_auc_score)
        #print("time:", finish_time)
        #print("\n")
        
    
    #print("--\n")
    
    return resultsList
    

def write(best_result, dataset_name):
    
    previous_result = read(dataset_name)
    
    print("\n")
    print("best_result     :", float(best_result.f1_score))
    print("previous_result :", float(previous_result))
    print("\n")
    
    if float(best_result.f1_score) <= float(previous_result):
        return False
    
    str_balancing = string_balancing(best_result.balancing)
    
    # #w - write and replace  #a - append
    with open(sys.path[0] + '/output/results.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([best_result.dataset_name, str_balancing + best_result.algorithm])
        writer.writerow(["accuracy_score", str(best_result.accuracy)])
        writer.writerow(["f1_score", str(best_result.f1_score)])
        writer.writerow(["roc_auc_score", str(best_result.roc_auc_score)])
        writer.writerow(["time", str(best_result.time)])
        writer.writerow(["---"])
    
    return True
        

def read(dataset_name):
    #dataset_name = "page-blocks0.dat" + ","
    dataset_name = dataset_name + ","
    selected_metric = "f1_score" + ","
    result = 0

    with open(sys.path[0] + '/output/results.csv') as f:

        i=0
        j=5 #4 metrics + 1 seperator
        found = False
        
        for line in f:
            if i % (j+1) == 0:
                if line.startswith(dataset_name):
                    found = True
                else:
                    found = False
            elif found and line.startswith(selected_metric):
                result = line.partition(selected_metric)[2]
            i+=1
        
    #returns the last result founded
    return result
    
    
def find_best_result(resultsList):
    return max(resultsList, key=lambda Results: Results.f1_score)


def string_balancing(balancing):
    str_balancing = ""
    if balancing != "-":
        str_balancing = balancing + " _ "
    return str_balancing
    

class Results(object):
    def __init__(self, dataset_name, balancing, algorithm, time, accuracy, f1_score, roc_auc_score):
        self.dataset_name = dataset_name
        self.balancing = balancing
        self.algorithm = algorithm
        self.time = time
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.roc_auc_score = roc_auc_score

