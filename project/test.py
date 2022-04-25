print("inicio")


#example ML

import sys
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
df = pd.read_csv(sys.path[0] + "/input/" + "page-blocks0.dat")

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
    
# print(list(df.columns))
# print(list(X.columns))
# print(list(y.columns))

# print(y.value_counts())

minimum_samples = min(y.value_counts())
if minimum_samples >= 5:
    minimum_samples = 5
else:
    minimum_samples -= 1
    
smote = SMOTE(random_state=42, k_neighbors=minimum_samples)
X, y = smote.fit_resample(X, y)

#print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()

algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced').fit(X_train, y_train.values.ravel())

finish_time = (round(time.time() - start_time,3))

algorithm_pred = algorithm.predict(X_train)

print("train")
print("metric_accuracy:         ", round(accuracy_score(y_train, algorithm_pred),3))
print("metric_f1_score          ", round(f1_score(y_train, algorithm_pred),3))
print("metric_roc_auc_score:    ", round(roc_auc_score(y_train, algorithm_pred),3))

algorithm_pred = algorithm.predict(X_test)

print("test")
print("metric_accuracy:         ", round(accuracy_score(y_test, algorithm_pred),3))
print("metric_f1_score          ", round(f1_score(y_test, algorithm_pred),3))
print("metric_roc_auc_score:    ", round(roc_auc_score(y_test, algorithm_pred),3))
print("time:                    ", finish_time)





#new write

#import sys
##import csv
#
#reading_file = open(sys.path[0] + '/output/results.csv', "r")
#
#new_file_content = ""
#i = 0
#j = 5           # 4 metrics + 1 seperator
#interval = 7-1  # if line to start is 7
#
#for line in reading_file:
#    stripped_line = line.strip()
#    
#    if interval <= i < (interval + j):
#        new_line = stripped_line.replace(stripped_line, "new string")
#    else:
#        new_line = stripped_line
#        
#    new_file_content += new_line +"\n"
#    i+=1
#reading_file.close()
#
#writing_file = open(sys.path[0] + '/output/results.csv', "w")
#writing_file.write(new_file_content)
#writing_file.close()





#previous write

## #w - write and replace  #a - append
#with open(sys.path[0] + '/output/results.csv', 'a', newline='') as f:
#    writer = csv.writer(f)
#
#    writer.writerow([best_result.dataset_name, str_balancing + best_result.algorithm])
#    writer.writerow(["accuracy_score", str(best_result.accuracy)])
#    writer.writerow(["f1_score", str(best_result.f1_score)])
#    writer.writerow(["roc_auc_score", str(best_result.roc_auc_score)])
#    writer.writerow(["time", str(best_result.time)])
#    writer.writerow(["---"])





print("fim")