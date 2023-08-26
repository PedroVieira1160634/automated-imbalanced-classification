from ml import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# NOTE: Make sure that the outcome column is labeled 'class' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
df, dataset_name = read_file_openml(450)

# features = tpot_data.drop('class', axis=1)
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['class'], random_state=42)

# Average CV score on the training set was: 0.9848245614035087
exported_pipeline = GaussianNB()

# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)


X, y, df_characteristics = features_labels(df, dataset_name)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scoring = {
    'balanced_accuracy': 'balanced_accuracy',
    'f1': 'f1', 
    'roc_auc': 'roc_auc',
    'g_mean': make_scorer(geometric_mean_score, greater_is_better=True),
    'cohen_kappa': make_scorer(cohen_kappa_score, greater_is_better=True)
    }
#scoring=scoring
scores = cross_validate(exported_pipeline, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1) #, return_train_score=True

# exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)

# balanced_accuracy = round(balanced_accuracy_score(testing_target, results),3)
# print("balanced_accuracy: ", balanced_accuracy)

balanced_accuracy_scores = scores['test_balanced_accuracy']
f1_score_scores = scores['test_f1']
roc_auc_score_scores = scores['test_roc_auc']
g_mean_score_scores = scores['test_g_mean']
cohen_kappa_scores = scores['test_cohen_kappa']

final_score_values = []
final_score_std_values = []

for i, (balanced_accuracy_value, f1_score_value, roc_auc_score_value, g_mean_score_value, cohen_kappa_value) in enumerate(zip(balanced_accuracy_scores, f1_score_scores, roc_auc_score_scores, g_mean_score_scores, cohen_kappa_scores)):
    final_score_value = round(np.mean([balanced_accuracy_value, f1_score_value, roc_auc_score_value, g_mean_score_value, cohen_kappa_value]), 3)
    final_score_values.append(final_score_value)
    
    final_score_std_value = round(np.std([balanced_accuracy_value, f1_score_value, roc_auc_score_value, g_mean_score_value, cohen_kappa_value]), 3)
    final_score_std_values.append(final_score_std_value)
    
    print(f"Fold {i+1}: balanced_accuracy = {round(balanced_accuracy_value, 3)}, f1_score = {round(f1_score_value, 3)}, roc_auc_score = {round(roc_auc_score_value, 3)}, g_mean_score = {round(g_mean_score_value, 3)}, cohen_kappa = {round(cohen_kappa_value, 3)}, final score = {final_score_value}, final score std = {final_score_std_value}")


balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
f1_score = round(np.mean(scores['test_f1']),3)
roc_auc_score = round(np.mean(scores['test_roc_auc']),3)
g_mean_score = round(np.mean(scores['test_g_mean']),3)
cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)

print("\nbalanced accuracy :", balanced_accuracy)
print("f1 score :", f1_score)
print("roc auc  :", roc_auc_score)
print("geometric mean score :", g_mean_score)
print("cohen kappa  :", cohen_kappa)
