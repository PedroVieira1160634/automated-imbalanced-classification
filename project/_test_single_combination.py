from ml import *

def classify_evaluate2(X, y, balancing, balancing_technique, dataset_name):

    #MUDAR AQUI
    array_classifiers = [
        # LGBMClassifier(random_state=42, objective='binary', class_weight='balanced', n_jobs=-1)
        # ,XGBClassifier(random_state=42, use_label_encoder=False, objective='binary:logistic', eval_metric='logloss', n_jobs=-1) #eval_metric=f1_score ; gpu; gpu_predictor
        # ,
        GradientBoostingClassifier(random_state=42)
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






# df, dataset_name = read_file(dataset_location)

id_openml = '37'
df, dataset_name = read_file_openml(id_openml)




start_time = time.time()
        
X, y, df_characteristics = features_labels(df, dataset_name)

#MUDAR AQUI
array_balancing = [
    "RandomOverSampler"
]

resultsList = []
i = 1
for balancing in array_balancing:
    try:
        print("loading: ", i, " of ", len(array_balancing))
        i += 1
        balancing_technique = pre_processing(balancing) 
        resultsList += classify_evaluate2(X, y, balancing, balancing_technique, dataset_name)
    except Exception:
        traceback.print_exc()

finish_time = (round(time.time() - start_time,3))

best_result = find_best_result(resultsList)

result_updated = write_results(best_result, finish_time)

write_full_results(resultsList, dataset_name)

write_characteristics(df_characteristics, best_result, result_updated)

