from ml import *

dataset_name = "glass1.dat"
df, dataset_name = read_file(sys.path[0] + "/input/" + dataset_name)
#df, dataset_name = read_file_openml(id)

X, y, characteristics = features_labels(df, dataset_name)

#write_characteristics(characteristics)

array_balancing = ["-","RandomUnderSampler","RandomOverSampler","SMOTE"]
resultsList = []

for balancing in array_balancing:
    X2, y2 = pre_processing(X, y, balancing) 
    resultsList += classify_evaluate(X2, y2, balancing, dataset_name)


#best_result = max(resultsList, key=lambda Results: Results.f1_score)

scores = []
for result in resultsList:
    scores.append(np.mean([result.balanced_accuracy, result.f1_score, result.roc_auc_score, result.g_mean_score, result.cohen_kappa_score]))

best_score = max(scores)
index = scores.index(best_score)
best_result = resultsList[index]


print(best_result.algorithm)
print(best_result.balancing)
print("time:                ", best_result.time)
print("balanced_accuracy:   ", best_result.balanced_accuracy)
print("f1_score:            ", best_result.f1_score)
print("roc_auc_score:       ", best_result.roc_auc_score)
print("g_mean_score:        ", best_result.g_mean_score)
print("cohen_kappa_score:   ", best_result.cohen_kappa_score)
