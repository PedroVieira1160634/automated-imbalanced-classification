#other ML tests

from ml import *
from datetime import datetime

print('\n\n----------------------------------start -', datetime.now(), '--------------------------------------\n\n')

dataset_name = "glass1.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)
#df, dataset_name = read_file_openml(id)

X, y, characteristics = features_labels(df, dataset_name)

#write_characteristics(characteristics)

# array_balancing = ["-", "SMOTE"]
array_balancing = [
        "-", 
        "ClusterCentroids", "CondensedNearestNeighbour", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "AllKNN", "InstanceHardnessThreshold", "NearMiss", "NeighbourhoodCleaningRule", "OneSidedSelection", "RandomUnderSampler", "TomekLinks",
        "RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "KMeansSMOTE", "SVMSMOTE",
        "SMOTEENN", "SMOTETomek"
    ]
resultsList = []

for balancing in array_balancing:
    try:
        X2, y2 = pre_processing(X, y, balancing) 
        resultsList += classify_evaluate(X2, y2, balancing, dataset_name)
    except Exception as e:
        print(e)




def write_full_results(resultsList):

    print("Write Full Results")
    
    df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "full_results.csv", sep=",")
    
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

    df_kb_r.to_csv(sys.path[0] + "/output/" + "full_results.csv", sep=",", index=False)
    
    print("File written, rows added!","\n")  



write_full_results(resultsList)





print('\n\n----------------------------------finish -', datetime.now(), '--------------------------------------\n\n')
