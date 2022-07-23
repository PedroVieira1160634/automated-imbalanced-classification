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


def execute_ml(dataset_location, id_openml, results_file_name, full_results_file_name, characteristics_file_name):
    
    try:
        if dataset_location:
            df, dataset_name = read_file(dataset_location)
        elif id_openml:
            df, dataset_name = read_file_openml(id_openml)
        else:
            return False
        
        if(not results_file_name):
            results_file_name = "kb_results"
        if(not full_results_file_name):
            full_results_file_name = "kb_full_results"
        if(not characteristics_file_name):
            characteristics_file_name = "kb_characteristics"
        
        start_time = time.time()
        
        X, y, df_characteristics = features_labels(df, dataset_name)
        
        # array_balancing = ["(no pre processing)"]
        # array_balancing = [
        #     "(no pre processing)", 
        #     "ClusterCentroids", "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN", "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler", "TomekLinks",
        #     "RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "SVMSMOTE",
        #     "SMOTEENN", "SMOTETomek"
        # ]
        array_balancing = [
            "RandomOverSampler", "SMOTE", "SVMSMOTE",
            "SMOTETomek"
        ]
        
        resultsList = []
        i = 1
        for balancing in array_balancing:
            try:
                print("loading: ", i, " of ", len(array_balancing))
                i += 1
                balancing_technique = pre_processing(balancing) 
                resultsList += classify_evaluate(X, y, balancing, balancing_technique, dataset_name)
            except Exception:
                traceback.print_exc()
        
        finish_time = (round(time.time() - start_time,3))
        
        best_result = find_best_result(resultsList)
        
        result_updated = write_results(best_result, finish_time, "kb_results")
        
        write_full_results(resultsList, dataset_name, "kb_full_results")
        
        write_characteristics(df_characteristics, best_result, result_updated, "kb_characteristics")
        
        return dataset_name
    
    except Exception:
        traceback.print_exc()
        return False



#  TEST VERSION
def execute_ml_test(dataset_location, id_openml):
    
    try:
        start_time = time.time()
        
        if dataset_location:
            df, dataset_name = read_file(dataset_location)
        elif id_openml:
            df, dataset_name = read_file_openml(id_openml)
        else:
            return False
        
        X, y, df_characteristics = features_labels(df, dataset_name)
        
        print("features_labels done!")
        
        #  TEST VERSION
        
        array_balancing = ["(no pre processing)"]
        resultsList = []
        for balancing in array_balancing:
            try:
                balancing_technique = pre_processing(balancing) 
                resultsList += classify_evaluate(X, y, balancing, balancing_technique, dataset_name)
            except Exception:
                traceback.print_exc()
        
        #  TEST VERSION
        
        finish_time = (round(time.time() - start_time,3))
        
        best_result = find_best_result(resultsList)

        current_value = round(np.mean([best_result.balanced_accuracy, best_result.f1_score, best_result.roc_auc_score, best_result.g_mean_score, best_result.cohen_kappa_score]), 3)
        elapsed_time = str(datetime.timedelta(seconds=round(finish_time,0)))
        
        print("Best Final Score Obtained    :", current_value)
        print("Elapsed Time                 :", elapsed_time, "\n")
        
        #  TEST VERSION
        
        return dataset_name
    
    except Exception:
        traceback.print_exc()
        return False



def execute_byCharacteristics(dataset_location, id_openml):
    try:
        if dataset_location:
            df, dataset_name = read_file(dataset_location)
        elif id_openml:
            df, dataset_name = read_file_openml(id_openml)
        else:
            return False
        
        X, y, df_characteristics = features_labels(df, dataset_name)
        
        first_row_to_remove = write_characteristics(df_characteristics, None, False, "kb_characteristics")
        
        if first_row_to_remove:
            df_dist = get_best_results_by_characteristics(dataset_name, "kb_characteristics")
            write_characteristics_remove_current_dataset("kb_characteristics")
            str_output = display_final_results(df_dist)
        else:
            str_output = "Attention: this dataset was already trained before!"
        
        return str_output
        
    except Exception:
        traceback.print_exc()
        return False



def read_file(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df, path.split('/')[-1]



def read_file_openml(id):
    
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe")

    df = pd.DataFrame(X, columns=attribute_names)
    df["class"] = y
    
    dataset_name = dataset.name + " (id:" + str(id) + ")"
    
    df = df.dropna()
    
    return df, dataset_name



def features_labels(df, dataset_name):
    
    print("\nDataset                      :", dataset_name, "\n")
    
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

    return X, y, df_characteristics



def pre_processing(balancing):
    
    balancing_technique = None
    
    # -- Under-sampling methods --
    if balancing == "ClusterCentroids":
        balancing_technique = ClusterCentroids(random_state=42)

    if balancing == "CondensedNearestNeighbour":
        balancing_technique = CondensedNearestNeighbour(random_state=42, n_jobs=-1)

    if balancing == "EditedNearestNeighbours":
        balancing_technique = EditedNearestNeighbours(n_jobs=-1)

    if balancing == "RepeatedEditedNearestNeighbours":
        balancing_technique = RepeatedEditedNearestNeighbours(n_jobs=-1)

    if balancing == "AllKNN":
        balancing_technique = AllKNN(n_jobs=-1)

    if balancing == "InstanceHardnessThreshold":
        balancing_technique = InstanceHardnessThreshold(random_state=42, n_jobs=-1)

    if balancing == "NearMiss":
        balancing_technique = NearMiss(n_jobs=-1)

    if balancing == "NeighbourhoodCleaningRule":
        balancing_technique = NeighbourhoodCleaningRule(n_jobs=-1)

    if balancing == "OneSidedSelection":
        balancing_technique = OneSidedSelection(random_state=42, n_jobs=-1)

    if balancing == "RandomUnderSampler":
        balancing_technique = RandomUnderSampler(random_state=42) #sampling_strategy=0.5
    
    if balancing == "TomekLinks":
        balancing_technique = TomekLinks(n_jobs=-1)
    
    
    # -- Over-sampling methods --
    if balancing == "RandomOverSampler":
        balancing_technique = RandomOverSampler(random_state=42) #sampling_strategy=0.5
    
    if balancing == "SMOTE":
        balancing_technique = SMOTE(random_state=42, n_jobs=-1) #sampling_strategy=0.5
    
    if balancing == "ADASYN":
        balancing_technique = ADASYN(random_state=42, n_jobs=-1)
    
    if balancing == "BorderlineSMOTE":
        balancing_technique = BorderlineSMOTE(random_state=42, n_jobs=-1)
    
    if balancing == "KMeansSMOTE":
        #UserWarning: MiniBatchKMeans
        # kmeans = MiniBatchKMeans(batch_size=2048)
        # , kmeans_estimator=kmeans
        
        # imbalance_ratio = 0
        # if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
        #     if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
        #         imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
        #     else:
        #         imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)
        
        # n_clusters = 1/imbalance_ratio
        
        balancing_technique = KMeansSMOTE(random_state=42, n_jobs=-1) #cluster_balance_threshold=n_clusters
    
    if balancing == "SVMSMOTE":
        balancing_technique = SVMSMOTE(random_state=42, n_jobs=-1)
    
    
    # -- Combination of over- and under-sampling methods --
    if balancing == "SMOTEENN":
        balancing_technique = SMOTEENN(random_state=42, n_jobs=-1)
        
    if balancing == "SMOTETomek":
        balancing_technique = SMOTETomek(random_state=42, n_jobs=-1)
    
    return balancing_technique



# initial:  1 + 19  balancing techniques and    11  classification algorithms   = 220   combinations
# second:   1 + 14  balancing techniques and    8   classification algorithms   = 120   combinations
# third:    12      balancing techniques and    6   classification algorithms   = 72    combinations
# fourth:   7       balancing techniques and    4   classification algorithms   = 28    combinations
# fifth:    5       balancing techniques and    3   classification algorithms   = 15    combinations
# final:    4       balancing techniques and    3   classification algorithms   = 12    combinations
def classify_evaluate(X, y, balancing, balancing_technique, dataset_name):

    array_classifiers = [
        # # LogisticRegression(random_state=42,max_iter=10000)
        # # ,GaussianNB() #no random_state (naive bayes)
        # # ,SVC(random_state=42)
        # # ,KNeighborsClassifier() #no random_state
        LGBMClassifier(random_state=42, objective='binary', class_weight='balanced', n_jobs=-1)
        ,XGBClassifier(random_state=42, use_label_encoder=False, objective='binary:logistic', eval_metric='logloss', n_jobs=-1) #eval_metric=f1_score ; gpu; gpu_predictor
        # # ,RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        # # ,ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        # # ,AdaBoostClassifier(random_state=42)
        # # ,BaggingClassifier(random_state=42, n_jobs=-1)
        ,GradientBoostingClassifier(random_state=42)
    ]
    
    resultsList = []
    
    for classifier in array_classifiers:
        start_time = time.time()
        
        model = make_pipeline(
            balancing_technique,
            classifier
        )
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        
        scoring = {
            'balanced_accuracy': 'balanced_accuracy',
            'f1': 'f1', 
            'roc_auc': 'roc_auc',
            'g_mean': make_scorer(geometric_mean_score, greater_is_better=True),
            'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True)
            }
        
        scores = cross_validate(model, X, y.values.ravel(), scoring=scoring,cv=cv, n_jobs=-1) #, return_train_score=True
        
        finish_time = round(time.time() - start_time,3)
        
        balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
        f1_score = round(np.mean(scores['test_f1']),3)
        roc_auc_score = round(np.mean(scores['test_roc_auc']),3)
        g_mean_score = round(np.mean(scores['test_g_mean']),3)
        cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)
        
        balanced_accuracy_std = round(np.std(scores['test_balanced_accuracy']),3)
        f1_score_std = round(np.std(scores['test_f1']),3)
        roc_auc_score_std = round(np.std(scores['test_roc_auc']),3)
        g_mean_score_std = round(np.std(scores['test_g_mean']),3)
        cohen_kappa_std = round(np.std(scores['test_cohen_kappa']),3)

        r1 = Results(dataset_name, balancing, classifier.__class__.__name__, finish_time, balanced_accuracy, balanced_accuracy_std, f1_score, f1_score_std, roc_auc_score, roc_auc_score_std, g_mean_score, g_mean_score_std, cohen_kappa, cohen_kappa_std)
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
    
    print("\nBest classifier is", best_result.algorithm, "with", string_balancing, "\n")
    
    return best_result



def write_characteristics(df_characteristics, best_result, result_updated, file_name):
    if df_characteristics.empty:
        print("--df_characteristics not valid on write_characteristics--")
        print("df_characteristics:", df_characteristics)
        return False
    
    #execute_byCharacteristics
    first_row_to_remove = True
    
    try:
    
        df_kb_c = pd.read_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",")
        #print(df_kb_c, '\n')
        
        df_kb_c_without = df_kb_c.loc[(df_kb_c["dataset"].values != df_characteristics["dataset"].values)]
        df_kb_c_selected = df_kb_c.loc[(df_kb_c["dataset"].values == df_characteristics["dataset"].values)]
        
        df_characteristics = pd.concat([df_characteristics, pd.DataFrame.from_records(df_kb_c_without)])
        df_characteristics = df_characteristics.reset_index(drop=True)
        
        #execute_ml
        if best_result and best_result.balancing and best_result.algorithm:
            #row updated or new line
            if result_updated or df_kb_c_selected.empty:
                df_characteristics.at[0, 'pre processing'] = best_result.balancing
                df_characteristics.at[0, 'algorithm'] = best_result.algorithm
                
                print("Characteristics written, row added or updated!","\n")
            
            #it was worse
            else:
                df_characteristics.at[0, 'pre processing'] = df_kb_c_selected["pre processing"].values[0]
                df_characteristics.at[0, 'algorithm'] = df_kb_c_selected["algorithm"].values[0]
                
                print("Characteristics not written!","\n")
        
        #execute_byCharacteristics
        else:
            #new row
            if df_kb_c_selected.empty:
                df_characteristics.at[0, 'pre processing'] = "?"
                df_characteristics.at[0, 'algorithm'] = "?"
            #remains value
            else:
                df_characteristics = df_kb_c
                first_row_to_remove = False
        
        df_characteristics.to_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",", index=False)
        
    except Exception:
        traceback.print_exc()
        return False

    return first_row_to_remove   



#writes if best
def write_results(best_result, elapsed_time, file_name):
    if not best_result:
        print("--best_result or elapsed_time not valid on write_results--")
        print("best_result:", best_result)
        print("elapsed_time:", elapsed_time)
        return False
    
    result_updated = False
    
    try:
        
        current_value = round(np.mean([best_result.balanced_accuracy, best_result.f1_score, best_result.roc_auc_score, best_result.g_mean_score, best_result.cohen_kappa_score]), 3)
        
        elapsed_time = str(datetime.timedelta(seconds=round(elapsed_time,0)))
        
        print("Best Final Score Obtained    :", current_value)
        print("Elapsed Time                 :", elapsed_time, "\n")
        
        df_kb_r = pd.read_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",")
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == best_result.dataset_name]
        
        if not df_kb_r2.empty :
            
            previous_value = round(np.mean([df_kb_r2['balanced accuracy'], df_kb_r2['f1 score'], df_kb_r2['roc auc'], df_kb_r2['geometric mean'], df_kb_r2['cohen kappa']]), 3)
            
            if current_value > previous_value:
                
                index = df_kb_r2.index.values[0]
                df_kb_r.at[index, 'pre processing'] = best_result.balancing
                df_kb_r.at[index, 'algorithm'] = best_result.algorithm
                df_kb_r.at[index, 'time'] = best_result.time
                df_kb_r.at[index, 'balanced accuracy'] = best_result.balanced_accuracy
                df_kb_r.at[index, 'balanced accuracy std'] = best_result.balanced_accuracy_std
                df_kb_r.at[index, 'f1 score'] = best_result.f1_score
                df_kb_r.at[index, 'f1 score std'] = best_result.f1_score_std
                df_kb_r.at[index, 'roc auc'] = best_result.roc_auc_score
                df_kb_r.at[index, 'roc auc std'] = best_result.roc_auc_score_std
                df_kb_r.at[index, 'geometric mean'] = best_result.g_mean_score
                df_kb_r.at[index, 'geometric mean std'] = best_result.g_mean_score_std
                df_kb_r.at[index, 'cohen kappa'] = best_result.cohen_kappa_score
                df_kb_r.at[index, 'cohen kappa std'] = best_result.cohen_kappa_score_std
                df_kb_r.at[index, 'total elapsed time'] = elapsed_time
                
                df_kb_r.to_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",", index=False)
                
                result_updated = True
                
                print("Results written, row updated!","\n")

            else:
                print("Results not written!","\n")
                
        else:
            
            df_kb_r.loc[len(df_kb_r.index)] = [
                best_result.dataset_name,
                best_result.balancing,
                best_result.algorithm,
                best_result.time,
                best_result.balanced_accuracy,
                best_result.balanced_accuracy_std,
                best_result.f1_score,
                best_result.f1_score_std,
                best_result.roc_auc_score,
                best_result.roc_auc_score_std,
                best_result.g_mean_score,
                best_result.g_mean_score_std,
                best_result.cohen_kappa_score,
                best_result.cohen_kappa_score_std,
                elapsed_time
            ]

            df_kb_r.to_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",", index=False)
            
            print("Results written, row added!","\n")  
        
    except Exception:
        traceback.print_exc()
        return False
    
    return result_updated



#only writes at first time 
def write_full_results(resultsList, dataset_name, file_name):
    if not resultsList or not dataset_name:
        print("--resultsList not valid on write_full_results--")
        print("resultsList:", resultsList)
        print("dataset_name:", dataset_name)
        return False
    
    try:
    
        df_kb_r = pd.read_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",")
        
        df_kb_r2 = df_kb_r.loc[df_kb_r['dataset'] == dataset_name]
        
        if df_kb_r2.empty :
        
            for result in resultsList:
                
                df_kb_r.loc[len(df_kb_r.index)] = [
                        result.dataset_name,
                        result.balancing,
                        result.algorithm,
                        result.time,
                        result.balanced_accuracy,
                        result.balanced_accuracy_std,
                        result.f1_score,
                        result.f1_score_std,
                        result.roc_auc_score,
                        result.roc_auc_score_std,
                        result.g_mean_score,
                        result.g_mean_score_std,
                        result.cohen_kappa_score,
                        result.cohen_kappa_score_std,
                        round(np.mean([result.balanced_accuracy, result.f1_score, result.roc_auc_score, result.g_mean_score, result.cohen_kappa_score]), 3)
                    ]

            df_kb_r.sort_values(by=['final score'], ascending=False, inplace=True)

            df_kb_r.to_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",", index=False)
            
            print("Full Results written, rows added!","\n")
        
        else:
            print("Full Results not written!","\n")
        
    except Exception:
        traceback.print_exc()
        return False
    
    return True



#by Euclidean Distance
def get_best_results_by_characteristics(dataset_name, file_name):
    if not dataset_name:
        print("--dataset_name not valid on get_best_results_by_characteristics--")
        print("best_result:", dataset_name)
        return False
    
    df_c = pd.read_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",")
    df_c = df_c.dropna(axis=1)
    df_c = df_c.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    df_c_a = df_c.loc[df_c['dataset'] == dataset_name]
    df_c_a = df_c_a.drop(['dataset', 'pre processing','algorithm'], axis=1)
    list_a = df_c_a.values.tolist()[0]
    list_a = [(float(i)-min(list_a))/(max(list_a)-min(list_a)) for i in list_a]

    df_c = df_c.loc[df_c['dataset'] != dataset_name]
    list_dist = []
    for index, row in df_c.iterrows():
        df_c_b = row.to_frame()
        df_c_b = df_c_b.drop(['dataset', 'pre processing','algorithm'])
        list_b = df_c_b.values.tolist()
        list_b = [x for xs in list_b for x in xs]
        list_b = [(float(i)-min(list_b))/(max(list_b)-min(list_b)) for i in list_b]
        list_dist.append((row['dataset'], row['pre processing'], row['algorithm'], np.linalg.norm(np.array(list_a) - np.array(list_b))))
        
    df_dist = pd.DataFrame(list_dist, columns=["dataset", "pre processing", "algorithm","result"])
    df_dist = df_dist.sort_values(by=['result'])
    df_dist = df_dist.drop_duplicates(subset=['pre processing', 'algorithm'], keep='first')
    df_dist = df_dist.reset_index(drop=True)
    df_dist = df_dist.head(3)
    
    print("Results:\n", df_dist)
    
    df_dist = df_dist[['pre processing', 'algorithm']]
    
    return df_dist



#always writes
def write_characteristics_remove_current_dataset(file_name):
    try:
        df_kb_c = pd.read_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",")
        df_kb_c = df_kb_c.iloc[1: , :]
        df_kb_c.to_csv(sys.path[0] + "/output/" + file_name + ".csv", sep=",", index=False)    
        # print("Removed Current Dataset Characteristics!","\n")
    
    except Exception:
        traceback.print_exc()
        return False



def display_final_results(df_dist):
    df_dist.loc[-1] = ['Pre Processing', 'Algorithm']
    df_dist.index = df_dist.index + 1
    df_dist = df_dist.sort_index()
    df_dist.insert(loc=0, column='rank', value=['Rank',1,2,3])
    
    str_output = "Top performing combinations of Pre Processing Technique with a Classifier Algorithm\n\n"
    str_output += "\n".join("{:7} {:25} {:25}".format(x, y, z) for x, y, z in zip(df_dist['rank'], df_dist['pre processing'], df_dist['algorithm']))
    str_output += "\n"
    return str_output



class Results(object):
    def __init__(self, dataset_name, balancing, algorithm, time, balanced_accuracy, balanced_accuracy_std, f1_score, f1_score_std, roc_auc_score, roc_auc_score_std, g_mean_score, g_mean_score_std, cohen_kappa_score, cohen_kappa_score_std):
        self.dataset_name = dataset_name
        self.balancing = balancing
        self.algorithm = algorithm
        self.time = time
        self.balanced_accuracy = balanced_accuracy
        self.balanced_accuracy_std = balanced_accuracy_std
        self.f1_score = f1_score
        self.f1_score_std = f1_score_std
        self.roc_auc_score = roc_auc_score
        self.roc_auc_score_std = roc_auc_score_std
        self.g_mean_score = g_mean_score
        self.g_mean_score_std = g_mean_score_std
        self.cohen_kappa_score = cohen_kappa_score
        self.cohen_kappa_score_std = cohen_kappa_score_std
