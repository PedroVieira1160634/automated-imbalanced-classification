import sys
import pandas as pd

def print_scores_pre_processing(df_kb_r):
    list_pre_processing = []

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "(no pre processing)"]
    # list_pre_processing.append(("(no pre processing)", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "ClusterCentroids"]
    # list_pre_processing.append(("ClusterCentroids", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "CondensedNearestNeighbour"]
    # list_pre_processing.append(("CondensedNearestNeighbour", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "EditedNearestNeighbours"]
    # list_pre_processing.append(("EditedNearestNeighbours", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "RepeatedEditedNearestNeighbours"]
    # list_pre_processing.append(("RepeatedEditedNearestNeighbours", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "AllKNN"]
    # list_pre_processing.append(("AllKNN", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "InstanceHardnessThreshold"]
    # list_pre_processing.append(("InstanceHardnessThreshold", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "NearMiss"]
    # list_pre_processing.append(("NearMiss", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "NeighbourhoodCleaningRule"]
    # list_pre_processing.append(("NeighbourhoodCleaningRule", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "OneSidedSelection"]
    # list_pre_processing.append(("OneSidedSelection", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "RandomUnderSampler"]
    # list_pre_processing.append(("RandomUnderSampler", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "TomekLinks"]
    # list_pre_processing.append(("TomekLinks", sum(df_kb_r2.index)))

    df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "RandomOverSampler"]
    list_pre_processing.append(("RandomOverSampler", sum(df_kb_r2.index)))

    df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SMOTE"]
    list_pre_processing.append(("SMOTE", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "ADASYN"]
    # list_pre_processing.append(("ADASYN", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "BorderlineSMOTE"]
    # list_pre_processing.append(("BorderlineSMOTE", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "KMeansSMOTE"]
    # list_pre_processing.append(("KMeansSMOTE", sum(df_kb_r2.index)))

    df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SVMSMOTE"]
    list_pre_processing.append(("SVMSMOTE", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SMOTEENN"]
    # list_pre_processing.append(("SMOTEENN", sum(df_kb_r2.index)))

    df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SMOTETomek"]
    list_pre_processing.append(("SMOTETomek", sum(df_kb_r2.index)))

    # print(list_pre_processing)
    df_pre_processing = pd.DataFrame(list_pre_processing, columns=['pre processing', 'score'])
    df_pre_processing = df_pre_processing.sort_values(by=['score'])

    # print(df_pre_processing.iloc[-15:])
    print(df_pre_processing)
    
    print("\nquantiles:\n", df_pre_processing["score"].quantile([0.25,0.5,0.75]))
    
def remove_worst_scores_pre_processing(df_kb_r):
    df_kb_r = df_kb_r.loc[
        (df_kb_r['pre processing'] != "SMOTEENN") & 
        (df_kb_r['pre processing'] != "CondensedNearestNeighbour") & 
        (df_kb_r['pre processing'] != "InstanceHardnessThreshold") &
        (df_kb_r['pre processing'] != "ClusterCentroids") &
        (df_kb_r['pre processing'] != "NearMiss") &
        
        (df_kb_r['pre processing'] != "OneSidedSelection") &
        (df_kb_r['pre processing'] != "(no pre processing)") &
        (df_kb_r['pre processing'] != "KMeansSMOTE") &
        
        (df_kb_r['pre processing'] != "EditedNearestNeighbours") &
        (df_kb_r['pre processing'] != "AllKNN") &
        (df_kb_r['pre processing'] != "RepeatedEditedNearestNeighbours") &
        (df_kb_r['pre processing'] != "NeighbourhoodCleaningRule") &
        (df_kb_r['pre processing'] != "RandomUnderSampler") &
        
        (df_kb_r['pre processing'] != "BorderlineSMOTE") &
        (df_kb_r['pre processing'] != "TomekLinks") &
        
        (df_kb_r['pre processing'] != "ADASYN")
        ]

    # df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",", index=False)
    
    print(df_kb_r)

def print_scores_classifier_algorithm(df_kb_r):
    list_classifier = []

    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "LogisticRegression"]
    # list_classifier.append(("LogisticRegression", sum(df_kb_r2.index)))
    
    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "GaussianNB"]
    # list_classifier.append(("GaussianNB", sum(df_kb_r2.index)))
    
    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "SVC"]
    # list_classifier.append(("SVC", sum(df_kb_r2.index)))
    
    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "KNeighborsClassifier"]
    # list_classifier.append(("KNeighborsClassifier", sum(df_kb_r2.index)))
    
    df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "LGBMClassifier"]
    list_classifier.append(("LGBMClassifier", sum(df_kb_r2.index)))

    df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "XGBClassifier"]
    list_classifier.append(("XGBClassifier", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "RandomForestClassifier"]
    # list_classifier.append(("RandomForestClassifier", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "ExtraTreesClassifier"]
    # list_classifier.append(("ExtraTreesClassifier", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "AdaBoostClassifier"]
    # list_classifier.append(("AdaBoostClassifier", sum(df_kb_r2.index)))

    # df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "BaggingClassifier"]
    # list_classifier.append(("BaggingClassifier", sum(df_kb_r2.index)))

    df_kb_r2 = df_kb_r.loc[df_kb_r['algorithm'] == "GradientBoostingClassifier"]
    list_classifier.append(("GradientBoostingClassifier", sum(df_kb_r2.index)))


    # print(list_classifier)
    df_classifier = pd.DataFrame(list_classifier, columns=['algorithm', 'score'])
    df_classifier = df_classifier.sort_values(by=['score'])

    # print(df_classifier.iloc[-5:])
    print(df_classifier)
    
    print("\nquantiles:\n", df_classifier["score"].quantile([0.25,0.5,0.75]))

def remove_worst_scores_classifier_algorithm(df_kb_r):
    df_kb_r = df_kb_r.loc[
        (df_kb_r['algorithm'] != "LogisticRegression") & 
        (df_kb_r['algorithm'] != "GaussianNB") &
        (df_kb_r['algorithm'] != "SVC") &
        
        (df_kb_r['algorithm'] != "AdaBoostClassifier") &
        (df_kb_r['algorithm'] != "KNeighborsClassifier") &
        
        (df_kb_r['algorithm'] != "BaggingClassifier") &
        (df_kb_r['algorithm'] != "ExtraTreesClassifier") &
        
        (df_kb_r['algorithm'] != "RandomForestClassifier")
        ]

    # df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",", index=False)

    print(df_kb_r)

#kb_full_results
#kb_results
#kb_characteristics
df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",")

print("\nDatasets on kb_full_results:\n", df_kb_r.dataset.value_counts(), "\n")
print("Number of datasets on kb_full_results:", df_kb_r.dataset.value_counts().count(), "\n")

print_scores_pre_processing(df_kb_r)
# remove_worst_scores_pre_processing(df_kb_r)

print_scores_classifier_algorithm(df_kb_r)
# remove_worst_scores_classifier_algorithm(df_kb_r)




# df_kb_r = df_kb_r.loc[(df_kb_r["dataset"].values == "pc2 (id:1069)")]
# # print(df_kb_r)

# # df_kb_r = df_kb_r[df_kb_r['f1 score']==df_kb_r['f1 score'].max()]
# # print(df_kb_r["f1 score"])

# df_kb_r = df_kb_r[df_kb_r['cohen kappa']==df_kb_r['cohen kappa'].max()]
# print(df_kb_r["cohen kappa"])




#pre_processing - to remove
#then, remove on ml.py


#1st
#SMOTEENN
#CondensedNearestNeighbour
#InstanceHardnessThreshold
#ClusterCentroids
#NearMiss

#2nd
#OneSidedSelection
#(no pre processing)
#KMeansSMOTE

#3rd
#EditedNearestNeighbours
#AllKNN
#RepeatedEditedNearestNeighbours
#NeighbourhoodCleaningRule
#RandomUnderSampler

#4th
#BorderlineSMOTE
#TomekLinks

#5th
#ADASYN


#classifier_algorithm - to remove

#1st
#LogisticRegression
#GaussianNB
#SVC

#2nd
#AdaBoostClassifier
#KNeighborsClassifier

#3rd
#BaggingClassifier
#ExtraTreesClassifier

#4th
#RandomForestClassifier

#5th
#(nothing)
