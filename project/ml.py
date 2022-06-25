import sys
import time
import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
import openml.datasets
from pymfe.mfe import MFE
from imblearn.pipeline import make_pipeline
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
import traceback
import warnings
warnings.filterwarnings("ignore")


def execute_ml(dataset_location, id_openml):
    
    try:
        start_time = time.time()
        
        if dataset_location:
            df, dataset_name = read_file(dataset_location)
        elif id_openml:
            df, dataset_name = read_file_openml(id_openml)
        else:
            return False
        
        X, y, df_characteristics = features_labels(df, dataset_name)
        
        array_balancing = ["-"]
        # array_balancing = ["-", "RandomUnderSampler", "RandomOverSampler", "SMOTE"]
        # array_balancing = [
        #     "-", 
        #     "ClusterCentroids", "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN", "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler", "TomekLinks",
        #     "RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "SVMSMOTE",
        #     "SMOTEENN", "SMOTETomek"
        # ]
        
        resultsList = []
        for balancing in array_balancing:
            try:
                X2, y2 = pre_processing(X, y, balancing) 
                resultsList += classify_evaluate(X2, y2, balancing, dataset_name)
            except Exception:
                traceback.print_exc()
        
        best_result = find_best_result(resultsList)
        
        write_characteristics(df_characteristics, best_result)
        
        write_full_results(resultsList, dataset_name)
        
        finish_time = (round(time.time() - start_time,3))
        write_results(best_result, finish_time)
        
        return dataset_name
    
    except Exception:
        traceback.print_exc()
        return False



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
    
    print("Dataset                      :", dataset_name, "\n")
    
    X = df.iloc[: , :-1]
    y = df.iloc[: , -1:]

    mfe = MFE(random_state=42, 
          groups=["complexity", "concept", "general", "itemset", "landmarking", "model-based", "statistical"], 
          summary=["mean", "sd", "kurtosis","skewness"])

    mfe.fit(X.values, y.values)
    ft = mfe.extract(suppress_warnings=True)
    
    df_characteristics = pd.DataFrame.from_records(ft)
    
    new_header = df_characteristics.iloc[0]
    df_characteristics = df_characteristics[1:]
    df_characteristics.columns = new_header
    
    df_characteristics.insert(loc=0, column="dataset", value=[dataset_name])
    
    
    encoded_columns = []
    for column_name in X.columns:
        if X[column_name].dtype == object or X[column_name].dtype.name == 'category':
            encoded_columns.extend([column_name])
    
    if encoded_columns:
        X = pd.get_dummies(X, X[encoded_columns].columns, drop_first=True)

    encoded_columns = []
    preserve_name = ""
    for column_name in y.columns:
        if y[column_name].dtype == object or y[column_name].dtype.name == 'category':
            encoded_columns.extend([column_name])
            preserve_name = column_name
    
    if encoded_columns:
        y = pd.get_dummies(y, y[encoded_columns].columns, drop_first=True)

    if preserve_name:
        y.rename(columns={y.columns[0]: preserve_name}, inplace = True)

    return X, y, df_characteristics



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
        # kmeans = MiniBatchKMeans(batch_size=2048)
        # , kmeans_estimator=kmeans
        
        imbalance_ratio = 0
        if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
            if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
                imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
            else:
                imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)
        
        n_clusters = 1/imbalance_ratio
        
        sm = KMeansSMOTE(random_state=42, n_jobs=-1, cluster_balance_threshold=n_clusters)
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



# initial: 1 + 19   balancing techniques and 7 classification algorithms    = 140   combinations
# final:   ?        balancing techniques and ? classification algorithms    = ?     combinations
def classify_evaluate(X, y, balancing, dataset_name):

    array_classifiers = [
        LogisticRegression(random_state=42,max_iter=10000)
        ,GaussianNB() #no random_state (naive bayes)
        ,SVC(random_state=42)
        ,KNeighborsClassifier() #no random_state
        ,LGBMClassifier(random_state=42, objective='binary', class_weight='balanced', n_jobs=-1)
        ,XGBClassifier(random_state=42, use_label_encoder=False, objective='binary:logistic', eval_metric='logloss', n_jobs=-1) #eval_metric=f1_score ; gpu; gpu_predictor
        ,RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        ,ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        ,AdaBoostClassifier(random_state=42)
        ,BaggingClassifier(random_state=42, n_jobs=-1)
        ,GradientBoostingClassifier(random_state=42)
    ]
    
    resultsList = []
    
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
        
    return resultsList



def find_best_result(resultsList):
    scores = []
    for result in resultsList:
        scores.append(np.mean([result.balanced_accuracy, result.f1_score, result.roc_auc_score, result.g_mean_score, result.cohen_kappa_score]))

    best_score = max(scores)
    index = scores.index(best_score)
    best_result = resultsList[index]
    
    string_balancing = best_result.balancing
    if string_balancing == "-":
        string_balancing = "no preprocessing"
    
    print("Best classifier is", best_result.algorithm, "with", string_balancing, "\n")
    
    return best_result



def write_characteristics(df_characteristics, best_result):

    if df_characteristics.empty or not best_result:
        print("--df_characteristics or best_result not valid on write_characteristics--")
        print("df_characteristics:", df_characteristics)
        print("best_result:", best_result)
        return False
    
    try:
    
        df_kb_c = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")
        #print(df_kb_c, '\n')
        
        df_kb_c = df_kb_c.loc[(df_kb_c["dataset"].values != df_characteristics["dataset"].values)]
        
        df_characteristics = df_characteristics.append(df_kb_c, ignore_index=True)
        
        df_characteristics.at[0, 'pre processing'] = best_result.balancing
        df_characteristics.at[0, 'algorithm'] = best_result.algorithm
        
        df_characteristics.to_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",", index=False)
        
        print("Write Characteristics written, row added or updated!","\n")
        
    except Exception:
        traceback.print_exc()
        return False

    return True   



def write_results(best_result, elapsed_time):

    if not best_result:
        print("--best_result or elapsed_time not valid on write_results--")
        print("best_result:", best_result)
        print("elapsed_time:", elapsed_time)
        return False
    
    try:
    
        current_value = round(np.mean([best_result.balanced_accuracy, best_result.f1_score, best_result.roc_auc_score, best_result.g_mean_score, best_result.cohen_kappa_score]), 3)
        
        elapsed_time = str(datetime.timedelta(seconds=round(elapsed_time,0)))
        
        df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == best_result.dataset_name]
        
        if not df_kb_r2.empty :
            
            previous_value = round(np.mean([df_kb_r2['balanced accuracy'], df_kb_r2['f1 score'], df_kb_r2['roc auc'], df_kb_r2['geometric mean'], df_kb_r2['cohen kappa']]), 3)
            
            if current_value > previous_value:
                
                index = df_kb_r2.index.values[0]
                df_kb_r.at[index, 'pre processing'] = best_result.balancing
                df_kb_r.at[index, 'algorithm'] = best_result.algorithm
                df_kb_r.at[index, 'time'] = best_result.time
                df_kb_r.at[index, 'balanced accuracy'] = best_result.balanced_accuracy
                df_kb_r.at[index, 'f1 score'] = best_result.f1_score
                df_kb_r.at[index, 'roc auc'] = best_result.roc_auc_score
                df_kb_r.at[index, 'geometric mean'] = best_result.g_mean_score
                df_kb_r.at[index, 'cohen kappa'] = best_result.cohen_kappa_score
                df_kb_r.at[index, 'total elapsed time'] = elapsed_time
                
                df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)
                
                print("Write Results written, row updated!","\n")

            else:
                print("Write Results not written!","\n")
                
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
                best_result.cohen_kappa_score,
                elapsed_time
            ]

            df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)
            
            print("Write Results written, row added!","\n")  
        
        print("Best Final Score Obtained    :", current_value)
        print("Elapsed Time                 :", elapsed_time, "\n")
        
    except Exception:
        traceback.print_exc()
        return False
    
    return True



def write_full_results(resultsList, dataset_name):

    if not resultsList or not dataset_name:
        print("--resultsList not valid on write_full_results--")
        print("resultsList:", resultsList)
        print("dataset_name:", dataset_name)
        return False
    
    try:
    
        df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",")
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == dataset_name]
        
        if df_kb_r2.empty :
        
            for result in resultsList:
                
                df_kb_r.loc[len(df_kb_r.index)] = [
                        result.dataset_name,
                        result.balancing,
                        result.algorithm,
                        result.time,
                        result.balanced_accuracy, 
                        result.f1_score, 
                        result.roc_auc_score,
                        result.g_mean_score,
                        result.cohen_kappa_score,
                        round(np.mean([result.balanced_accuracy, result.f1_score, result.roc_auc_score, result.g_mean_score, result.cohen_kappa_score]), 3)
                    ]

            df_kb_r.sort_values(by=['final score'], ascending=False, inplace=True)

            df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",", index=False)
            
            print("Write Full Results written, rows added!","\n")
        
        else:
            print("Write Full Results not written!","\n")
        
    except Exception:
        traceback.print_exc()
        return False
    
    return True



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
