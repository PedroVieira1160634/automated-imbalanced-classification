from ml import *

class ResultsSHT(object):
    def __init__(self, dataset_name, balancing, algorithm, output_app, time, final_score_values, final_score_std_values):
        self.dataset_name = dataset_name
        self.balancing = balancing
        self.algorithm = algorithm
        self.output_app = output_app
        self.time = time
        self.final_score_values = final_score_values
        self.final_score_std_values = final_score_std_values


def classify_evaluate2(X, y, balancing, balancing_technique, dataset_name):

    #MUDAR AQUI
    array_classifiers = [
        # LGBMClassifier(random_state=42, objective='binary', class_weight='balanced', n_jobs=-1)
        # XGBClassifier(random_state=42, use_label_encoder=False, objective='binary:logistic', eval_metric='logloss', n_jobs=-1) #eval_metric=f1_score ; gpu; gpu_predictor
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
        
        #scoring=scoring
        scores = cross_validate(model, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1) #, return_train_score=True
        
        finish_time = round(time.time() - start_time,3)
        
        # # Access the train_score and test_score for each fold
        # train_scores = scores['train_score']
        # test_scores = scores['test_score']

        # # Print the train and test scores for each fold
        # for i, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
        #     print(f"Fold {i+1}: Train Score = {train_score}, Test Score = {test_score}")
        
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
        
        
        # balanced_accuracy = round(np.mean(scores['test_balanced_accuracy']),3)
        # f1_score = round(np.mean(scores['test_f1']),3)
        # roc_auc_score = round(np.mean(scores['test_roc_auc']),3)
        # g_mean_score = round(np.mean(scores['test_g_mean']),3)
        # cohen_kappa = round(np.mean(scores['test_cohen_kappa']),3)
        
        # balanced_accuracy_std = round(np.std(scores['test_balanced_accuracy']),3)
        # f1_score_std = round(np.std(scores['test_f1']),3)
        # roc_auc_score_std = round(np.std(scores['test_roc_auc']),3)
        # g_mean_score_std = round(np.std(scores['test_g_mean']),3)
        # cohen_kappa_std = round(np.std(scores['test_cohen_kappa']),3)

        #MUDAR AQUI
        # output_app = 'LM'
        output_app = 'RM'
        
        print("")
        print("APP:", output_app)
        print("balancing:", balancing)
        print("classifier:", classifier.__class__.__name__)
        print("finish_time:", finish_time)
        
        r1 = ResultsSHT(dataset_name, balancing, classifier.__class__.__name__, output_app, finish_time, final_score_values, final_score_std_values)
        resultsList.append(r1)
        
        # r1 = Results(dataset_name, balancing, classifier.__class__.__name__, finish_time, balanced_accuracy, balanced_accuracy_std, f1_score, f1_score_std, roc_auc_score, roc_auc_score_std, g_mean_score, g_mean_score_std, cohen_kappa, cohen_kappa_std)
        # resultsList.append(r1)
        
    return resultsList


def write_results_sht(result):
    if not result:
        print("--result not valid on write_results--")
        print("result:", result)
        return False
    
    try:
        
        df_kb_r = pd.read_csv(application_path + "/output/" + "results_sht.csv", sep=",")
        
        df_kb_r2 = df_kb_r.loc[(df_kb_r['dataset'] == result.dataset_name) & (df_kb_r['output app'] == result.output_app)]
        
        if df_kb_r2.empty :
        
            fold = 1
            for final_score, final_score_std in zip(result.final_score_values, result.final_score_std_values):
                df_kb_r.loc[len(df_kb_r.index)] = [
                    result.dataset_name,
                    result.balancing,
                    result.algorithm,
                    result.output_app,
                    fold,
                    result.time,
                    final_score,
                    final_score_std
                ]
                fold += 1

            df_kb_r.to_csv(application_path + "/output/" + "results_sht.csv", sep=",", index=False)
            
            print("Results SHT written, row added!","\n")
        
        else:
            print("Results SHT not written!","\n")
        
    except Exception:
        traceback.print_exc()
        return False
    
    return True





# df, dataset_name = read_file(dataset_location)

#MUDAR AQUI
id_openml = '976'
df, dataset_name = read_file_openml(id_openml)
# df, dataset_name = read_file(sys.path[0] + "/input/" + "kr-vs-k-zero_vs_eight.dat")



start_time = time.time()
        
X, y, df_characteristics = features_labels(df, dataset_name)

#MUDAR AQUI
# "RandomOverSampler", "SMOTE", "SVMSMOTE", "SMOTETomek"
array_balancing = [
    "RandomOverSampler"
]

resultsList = []
i = 1
for balancing in array_balancing:
    try:
        # print("loading: ", i, " of ", len(array_balancing))
        balancing_technique = pre_processing(balancing) 
        resultsList += classify_evaluate2(X, y, balancing, balancing_technique, dataset_name)
        write_results_sht(resultsList[i-1])
        i += 1
    except Exception:
        traceback.print_exc()


# finish_time = (round(time.time() - start_time,3))

# best_result = find_best_result(resultsList)

# result_updated = write_results(best_result, finish_time)

# write_full_results(resultsList, dataset_name)

# write_characteristics(df_characteristics, best_result, result_updated)
