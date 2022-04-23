print("inicio")


#example ML

import sys
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
df = pd.read_csv(sys.path[0] + "/input/" + "kddcup-rootkit-imap_vs_back.dat")


encoded_columns = []
for column_name in df.columns:
    if df[column_name].dtype == object:
        encoded_columns.extend([column_name])
    else:
        pass

df = pd.get_dummies(df, df[encoded_columns].columns, drop_first=True)


X = df.iloc[:,:-1]
y = df.iloc[:,-1:]

#SMOTE
oversample = SMOTE(random_state=42)
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()

#ExtraTreesClassifier
algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced').fit(X_train, y_train.values.ravel())

finish_time = (round(time.time() - start_time,3))

algorithm_pred = algorithm.predict(X_test)

print("metric_accuracy:         ", accuracy_score(y_test, algorithm_pred))
print("metric_f1_score          ", f1_score(y_test, algorithm_pred))
print("metric_roc_auc_score:    ", roc_auc_score(y_test, algorithm_pred))
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