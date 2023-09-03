from ml import *

def write_results_sht_wilcoxon(dataset, test_type, stat, p, res):
    if not dataset or not test_type: #or not stat or not p or not res:
        print("--result not valid on write_results_sht_wilcoxon--")
        print("dataset:", dataset)
        print("test_type:", test_type)
        print("stat:", stat)
        print("p:", p)
        print("res:", res)
        return False
    
    try:
        
        df_kb_r = pd.read_csv(application_path + "/output/" + "results_sht_wilcoxon.csv", sep=",")
        
        df_kb_r2 = df_kb_r.loc[(df_kb_r['dataset'] == dataset) & (df_kb_r['test type'] == test_type)]
        
        if df_kb_r2.empty:
            df_kb_r.loc[len(df_kb_r.index)] = [
                dataset,
                test_type,
                stat,
                p,
                res
            ]
            
            df_kb_r.to_csv(application_path + "/output/" + "results_sht_wilcoxon.csv", sep=",", index=False)

            print("Results SHT Wilcoxon written, row added!", "\n")

        else:
            print("Results SHT Wilcoxon not written!", "\n")
            
    except Exception:
        traceback.print_exc()
        return False
    
    return True



# Wilcoxon signed-rank test
# from numpy.random import seed
# from numpy.random import randn
from scipy.stats import wilcoxon

# seed the random number generator
# seed(1)
# # generate two independent samples
# data1 = 5 * randn(100) + 50
# data2 = 5 * randn(100) + 51

# # other example
# data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
# data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]


# #learning module
# df_lm = pd.read_csv(application_path + "/output/" + "kb_results.csv", sep=",")
# df_lm = df_lm.loc[
#     (df_lm['dataset'] == 'dis (id:40713)') | 
#     (df_lm['dataset'] == 'musk (id:1116)') |
#     (df_lm['dataset'] == 'mfeat-fourier (id:971)') |
#     (df_lm['dataset'] == 'Satellite (id:40900)') |
#     (df_lm['dataset'] == 'arsenic-male-bladder (id:947)') |
#     (df_lm['dataset'] == 'analcatdata_apnea2 (id:765)') |
#     (df_lm['dataset'] == 'regime_alimentaire (id:42172)') |
#     (df_lm['dataset'] == 'page-blocks0.dat') |
#     (df_lm['dataset'] == 'dgf_test (id:42883)') |
#     (df_lm['dataset'] == 'cpu_small (id:735)') |
#     (df_lm['dataset'] == 'analcatdata_birthday (id:968)') |
#     (df_lm['dataset'] == 'optdigits (id:980)') |
#     (df_lm['dataset'] == 'kr-vs-k-zero_vs_eight.dat') |
#     (df_lm['dataset'] == 'analcatdata_lawsuit (id:450)') |
#     (df_lm['dataset'] == 'JapaneseVowels (id:976)')
#     ]
# # Reorder the DataFrame
# custom_order = ["14", "15", "5", "3", "10", "9", "11", "7", "8", "6", "4", "12", "13", "1", "2"]
# df_lm["order"] = custom_order
# df_lm["order"] = df_lm["order"].astype('Int32')
# df_lm.sort_values(by=["order"], inplace=True)
# # print(df_lm)

# # values_lm = df_lm[['balanced accuracy', 'f1 score', 'roc auc', 'geometric mean', 'cohen kappa']].to_numpy().flatten() #.flatten()
# values_lm = df_lm['roc auc'].to_numpy()
# print("Learning Module metrics: ", values_lm)


# #recommendation module
# df_rm = pd.read_csv(application_path + "/output/" + "kb_full_results.csv", sep=",")
# df_rm = df_rm.loc[
#     ( (df_rm['dataset'] == 'dis (id:40713)') & (df_rm['pre processing'] == 'RandomOverSampler') & (df_rm['algorithm'] == 'LGBMClassifier') ) |
#     ( (df_rm['dataset'] == 'musk (id:1116)') & (df_rm['pre processing'] == 'RandomOverSampler') & (df_rm['algorithm'] == 'XGBClassifier') ) |
#     ( (df_rm['dataset'] == 'mfeat-fourier (id:971)') & (df_rm['pre processing'] == 'SVMSMOTE') & (df_rm['algorithm'] == 'GradientBoostingClassifier') ) |
#     ( (df_rm['dataset'] == 'Satellite (id:40900)') & (df_rm['pre processing'] == 'SMOTE') & (df_rm['algorithm'] == 'GradientBoostingClassifier') ) |
#     ( (df_rm['dataset'] == 'arsenic-male-bladder (id:947)') & (df_rm['pre processing'] == 'RandomOverSampler') & (df_rm['algorithm'] == 'LGBMClassifier') ) |
#     ( (df_rm['dataset'] == 'analcatdata_apnea2 (id:765)') & (df_rm['pre processing'] == 'RandomOverSampler') & (df_rm['algorithm'] == 'GradientBoostingClassifier') ) |
#     ( (df_rm['dataset'] == 'regime_alimentaire (id:42172)') & (df_rm['pre processing'] == 'SVMSMOTE') & (df_rm['algorithm'] == 'XGBClassifier') ) |
#     ( (df_rm['dataset'] == 'page-blocks0.dat') & (df_rm['pre processing'] == 'SMOTE') & (df_rm['algorithm'] == 'LGBMClassifier') ) |
#     ( (df_rm['dataset'] == 'dgf_test (id:42883)') & (df_rm['pre processing'] == 'SVMSMOTE') & (df_rm['algorithm'] == 'LGBMClassifier') ) |
#     ( (df_rm['dataset'] == 'cpu_small (id:735)') & (df_rm['pre processing'] == 'SMOTETomek') & (df_rm['algorithm'] == 'GradientBoostingClassifier') ) |
#     ( (df_rm['dataset'] == 'analcatdata_birthday (id:968)') & (df_rm['pre processing'] == 'RandomOverSampler') & (df_rm['algorithm'] == 'LGBMClassifier') ) |
#     ( (df_rm['dataset'] == 'optdigits (id:980)') & (df_rm['pre processing'] == 'RandomOverSampler') & (df_rm['algorithm'] == 'GradientBoostingClassifier') ) |
#     ( (df_rm['dataset'] == 'kr-vs-k-zero_vs_eight.dat') & (df_rm['pre processing'] == 'SVMSMOTE') & (df_rm['algorithm'] == 'XGBClassifier') ) |
#     ( (df_rm['dataset'] == 'analcatdata_lawsuit (id:450)') & (df_rm['pre processing'] == 'SVMSMOTE') & (df_rm['algorithm'] == 'GradientBoostingClassifier') ) |
#     ( (df_rm['dataset'] == 'JapaneseVowels (id:976)') & (df_rm['pre processing'] == 'RandomOverSampler') & (df_rm['algorithm'] == 'GradientBoostingClassifier') )
#     ]
# # Reorder the DataFrame
# custom_order = ["2", "3", "12", "9", "15", "13", "14", "8", "7", "10", "6", "4", "1", "11", "5"]
# df_rm["order"] = custom_order
# df_rm["order"] = df_rm["order"].astype('Int32')
# df_rm.sort_values(by=["order"], inplace=True)
# # print(df_rm)

# # values_rm = df_rm[['balanced accuracy', 'f1 score', 'roc auc', 'geometric mean', 'cohen kappa']].to_numpy().flatten() #.flatten()
# values_rm = df_rm['roc auc'].to_numpy()
# print("Recommendation Module metrics: ", values_rm)


# #TPOT
# df_tpot = pd.read_csv(application_path + "/output/" + "results_TPOT.csv", sep=",")
# # print(df_tpot)

# # values_tpot = df_tpot[['balanced accuracy', 'f1 score', 'roc auc', 'geometric mean', 'cohen kappa']].to_numpy().flatten() #.flatten()
# values_tpot = df_tpot['roc auc'].to_numpy()
# print("TPOT metrics: ", values_tpot)


df_kb_r = pd.read_csv(application_path + "/output/" + "results_sht.csv", sep=",")

#MUDAR AQUI
dataset = 'arsenic-male-bladder (id:947)'

#internal
# df_kb_r_1 = df_kb_r.loc[(df_kb_r['dataset'] == dataset) & (df_kb_r['output app'] == 'LM')]
# df_kb_r_2 = df_kb_r.loc[(df_kb_r['dataset'] == dataset) & (df_kb_r['output app'] == 'RM')]

# #external
df_kb_r_1 = df_kb_r.loc[(df_kb_r['dataset'] == dataset) & (df_kb_r['output app'] == 'RM')]
df_kb_r_2 = df_kb_r.loc[(df_kb_r['dataset'] == dataset) & (df_kb_r['output app'] == 'TPOT')]

df_kb_r_1 = df_kb_r_1['final score'].to_numpy()
df_kb_r_2 = df_kb_r_2['final score'].to_numpy()

#internal
# print("\nLearning Module Final Score by Fold: ", df_kb_r_1)
# print("\nRecommendation Module Final Score by Fold: ", df_kb_r_2)

#external
print("\nRecommendation Module Final Score by Fold: ", df_kb_r_1)
print("\nTPOT Final Score by Fold: ", df_kb_r_2)


data1 = df_kb_r_1
data2 = df_kb_r_2

# compare samples
stat, p = wilcoxon(data1, data2)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
print("\nStatistics=", stat)
print("p=", p)
# interpret
alpha = 0.05
res=''

print("")
if p > alpha:
	res = 'Same distribution (fail to reject H0)'
	print(res)
else:
	res = 'Different distribution (reject H0)'
	print(res)
print("")

# test_type = 'internal'
test_type = 'external'
write_results_sht_wilcoxon(dataset, test_type, stat, p, res)





# import numpy as np
# from scipy.stats import friedmanchisquare

# # Example evaluation metrics for three algorithms on three datasets
# algorithm1_metrics = np.array([[0.85, 0.92, 0.78, 0.91],
#                                [0.75, 0.88, 0.72, 0.85],
#                                [0.91, 0.94, 0.83, 0.88]])

# algorithm2_metrics = np.array([[0.88, 0.89, 0.79, 0.92],
#                                [0.78, 0.82, 0.75, 0.88],
#                                [0.92, 0.95, 0.87, 0.91]])

# algorithm3_metrics = np.array([[0.72, 0.81, 0.68, 0.79],
#                                [0.65, 0.72, 0.61, 0.75],
#                                [0.75, 0.79, 0.68, 0.72]])

# # Perform the Friedman test
# statistic, p_value = friedmanchisquare(algorithm1_metrics, algorithm2_metrics, algorithm3_metrics)

# print("Friedman test statistic:", statistic)
# print("P-value:", p_value)

# # Check if the p-value is significant (common significance level of 0.05)
# if p_value < 0.05:
#     print("Reject the null hypothesis: There are significant differences among algorithms.")
# else:
#     print("Fail to reject the null hypothesis: There are no significant differences among algorithms.")
