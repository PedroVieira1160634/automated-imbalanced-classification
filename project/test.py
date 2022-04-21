print("inicio")


#example ML

import sys
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
df = pd.read_csv(sys.path[0] + "/input/" + "page-blocks0.dat")

y = df.iloc[:,-1:]
X = df.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#print(x_train)
#print(x_test)

#ver isto melhor

#Label Encoding vs OneHot Encoding vs get_dummies em input e output
#https://stackoverflow.com/questions/30384995/randomforestclassfier-fit-valueerror-could-not-convert-string-to-float

# le = LabelEncoder()

# for column_name in x_train.columns:
#     if x_train[column_name].dtype == object:
#         x_train[column_name] = le.fit_transform(x_train[column_name])
#     else:
#         pass
    
# for column_name in x_test.columns:
#     if x_test[column_name].dtype == object:
#         x_test[column_name] = le.fit_transform(x_test[column_name])
#     else:
#         pass



# ohc = OneHotEncoder(handle_unknown='ignore')

# for column_name in x_train.columns:
#     if x_train[column_name].dtype == object:
#         x_train[column_name] = ohc.fit_transform(x_train[column_name])
#         #pd.DataFrame(enc.fit_transform(bridge_df[['Bridge_Types_Cat']]).toarray())
#     else:
#         pass


#print(x_train)
#print(x_test)



start_time = time.time()

algorithm = ExtraTreesClassifier(random_state=42, class_weight='balanced').fit(x_train, y_train.values.ravel()) 

finish_time = (round(time.time() - start_time,5))

algorithm_pred = algorithm.predict(x_test)

print("metric_accuracy:         ", round(accuracy_score(y_test, algorithm_pred),5))
print("metric_f1_score          ", round(f1_score(y_test, algorithm_pred),5))
print("metric_roc_auc_score:    ", round(roc_auc_score(y_test, algorithm_pred),5))
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