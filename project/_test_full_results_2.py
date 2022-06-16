import sys
import pandas as pd

df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "full_results.csv", sep=",")

# df_kb_r = df_kb_r.loc[df_kb_r['dataset'] == "glass1.dat"]
# print(df_kb_r)

#pre processing
list_pre_processing = []

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "-"]
list_pre_processing.append(("-", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "ClusterCentroids"]
list_pre_processing.append(("ClusterCentroids", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "CondensedNearestNeighbour"]
list_pre_processing.append(("CondensedNearestNeighbour", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "EditedNearestNeighbours"]
list_pre_processing.append(("EditedNearestNeighbours", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "RepeatedEditedNearestNeighbours"]
list_pre_processing.append(("RepeatedEditedNearestNeighbours", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "AllKNN"]
list_pre_processing.append(("AllKNN", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "InstanceHardnessThreshold"]
list_pre_processing.append(("InstanceHardnessThreshold", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "NearMiss"]
list_pre_processing.append(("NearMiss", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "NeighbourhoodCleaningRule"]
list_pre_processing.append(("NeighbourhoodCleaningRule", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "OneSidedSelection"]
list_pre_processing.append(("OneSidedSelection", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "RandomUnderSampler"]
list_pre_processing.append(("RandomUnderSampler", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "TomekLinks"]
list_pre_processing.append(("TomekLinks", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "RandomOverSampler"]
list_pre_processing.append(("RandomOverSampler", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SMOTE"]
list_pre_processing.append(("SMOTE", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "ADASYN"]
list_pre_processing.append(("ADASYN", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "BorderlineSMOTE"]
list_pre_processing.append(("BorderlineSMOTE", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "KMeansSMOTE"]
list_pre_processing.append(("KMeansSMOTE", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SVMSMOTE"]
list_pre_processing.append(("SVMSMOTE", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SMOTEENN"]
list_pre_processing.append(("SMOTEENN", sum(df_kb_r2.index)))

df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "SMOTETomek"]
list_pre_processing.append(("SMOTETomek", sum(df_kb_r2.index)))



# print(list_pre_processing)
df_pre_processing = pd.DataFrame(list_pre_processing, columns=['pre processing', 'score'])

df_pre_processing.sort_values(by=['score'], inplace=True)

# print(df_pre_processing)
print(df_pre_processing.iloc[-15:])








# #classifier algorithm
# list_classifier = []

# df_kb_r2 = df_kb_r.loc[df_kb_r['pre processing'] == "-"]
# list_classifier.append(("-", sum(df_kb_r2.index)))

