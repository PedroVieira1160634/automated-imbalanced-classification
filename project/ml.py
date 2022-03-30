import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import csv


def read_file(path):
    return pd.read_csv(path)


def train_test_split_func(df):
    Y = df.iloc[:,-1:]
    X = df.iloc[:,:-1]

    return train_test_split(X, Y, test_size=0.2, random_state=42)


def classify_evaluate_write(x_train, x_test, y_train, y_test, dataset_name):

    array_classifiers = [lgb.LGBMClassifier(), XGBClassifier(), RandomForestClassifier()]

    resultsList = []
    
    for classifier in array_classifiers:
        
        start_time = time.time()
        
        algorithm = classifier.fit(x_train, y_train)

        finish_time = (round(time.time() - start_time,5))

        algorithm_pred = algorithm.predict(x_test)

        metric_accuracy = round(accuracy_score(y_test, algorithm_pred),5)
        metric_f1_score = round(f1_score(y_test, algorithm_pred),5)
        metric_roc_auc_score = round(roc_auc_score(y_test, algorithm_pred),5)
        
        r1 = Results(dataset_name, classifier.__class__.__name__, finish_time, metric_accuracy, metric_f1_score, metric_roc_auc_score)
        resultsList.append(r1)
        
        #print("\n",r1,"\n")
        
        #to uncomment
        # #w - write and replace
        # #a - append
        # with open(sys.path[0] + '/output/results.csv', 'a', newline='') as f:
        #     writer = csv.writer(f)

        #     writer.writerow([dataset_name, classifier.__class__.__name__])
        #     writer.writerow(["accuracy_score", str(metric_accuracy)])
        #     writer.writerow(["f1_score", str(metric_f1_score)])
        #     writer.writerow(["roc_auc_score", str(metric_roc_auc_score)])
        #     writer.writerow(["time", str(finish_time)])
        #     writer.writerow(["---"])
    
    return resultsList


def find_best_result(resultsList):
    return max(resultsList.f1_score)


class Results():
    def __init__(self, dataset_name, algorithm, time, accuracy, f1_score, roc_auc_score):
        self.dataset_name = dataset_name
        self.algorithm = algorithm
        self.time = time
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.roc_auc_score = roc_auc_score

