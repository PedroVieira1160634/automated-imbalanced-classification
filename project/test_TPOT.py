
#test TOPT

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tpot import TPOTClassifier

#glass1.dat
#page-blocks0.dat
#kddcup-rootkit-imap_vs_back.dat
df = pd.read_csv(sys.path[0] + "/input/" + "glass1.dat")

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
for column_name in y.columns:
    if y[column_name].dtype == object:
        encoded_columns.extend([column_name])
    else:
        pass

y = pd.get_dummies(y, y[encoded_columns].columns, drop_first=True)


smote = SMOTE(random_state=42) #, k_neighbors=minimum_samples
X, y = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#https://epistasislab.github.io/tpot/api/
pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, scoring='f1', cv=5, n_jobs=-1, random_state=42, verbosity=2, memory='auto')

pipeline_optimizer.fit(X_train, y_train.values.ravel())
print("score:", pipeline_optimizer.score(X_test, y_test.values.ravel()))
pipeline_optimizer.export('tpot_exported_pipeline.py')

#glass1.dat
#without preprocessing
#0.8125
#GradientBoostingClassifier         # without NN

#glass1.dat
#SMOTE
#0.923
#RandomForestClassifier



