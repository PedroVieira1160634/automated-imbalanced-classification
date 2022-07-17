import sys
import pandas as pd

def remove_all_worst_datasets():
    # #kb_results
    # df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")
    
    # df_kb_r = df_kb_r.loc[
    #     #low metrics
    #     (df_kb_r['dataset'] != "pc2 (id:1069)") &
    #     (df_kb_r['dataset'] != "thoracic-surgery (id:1506)") &
    #     (df_kb_r['dataset'] != "PieChart2 (id:1452)") &
    #     (df_kb_r['dataset'] != "ibm-employee-performance (id:43905)") &
        
    #     #repeated datasets
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43893)") &
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43894)") &
    #     (df_kb_r['dataset'] != "climate-model-simulation-crashes (id:1467)") &
    #     (df_kb_r['dataset'] != "MegaWatt1 (id:1442)") &
        
    #     #low metrics after
    #     (df_kb_r['dataset'] != "kc3 (id:1065)") &
    #     (df_kb_r['dataset'] != "kc1 (id:1067)") &
    #     (df_kb_r['dataset'] != "pc1 (id:1068)") &
    #     (df_kb_r['dataset'] != "PizzaCutter3 (id:1444)") &
    #     (df_kb_r['dataset'] != "pc3 (id:1050)") &
    #     (df_kb_r['dataset'] != "CostaMadre1 (id:1446)") &
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43896)") &
    #     (df_kb_r['dataset'] != "lungcancer_GSE31210 (id:1412)") &
    #     (df_kb_r['dataset'] != "PieChart3 (id:1453)") &
    #     (df_kb_r['dataset'] != "MeanWhile1 (id:1449)") &
    #     (df_kb_r['dataset'] != "PizzaCutter1 (id:1443)") &
    #     (df_kb_r['dataset'] != "mw1 (id:1071)") &
    #     (df_kb_r['dataset'] != "yeast_ml8 (id:316)") &
    #     (df_kb_r['dataset'] != "Speech (id:40910)") &
        
    #     #low metrics after part 2
    #     (df_kb_r['dataset'] != "haberman (id:43)") &
    #     (df_kb_r['dataset'] != "autoUniv-au1-1000 (id:1547)") &
    #     (df_kb_r['dataset'] != "balloon (id:914)") &
    #     (df_kb_r['dataset'] != "blood-transfusion-service-center (id:1464)") &
    #     (df_kb_r['dataset'] != "SPECT (id:336)") &
    #     (df_kb_r['dataset'] != "SPECTF (id:1600)") &
    #     (df_kb_r['dataset'] != "kc2 (id:1063)") &
    #     (df_kb_r['dataset'] != "analcatdata_dmft (id:1014)") &
    #     (df_kb_r['dataset'] != "solar-flare (id:40702)") &
    #     (df_kb_r['dataset'] != "mc1 (id:1056)") &
        
    #     #low metrics after part 3
    #     (df_kb_r['dataset'] != "Titanic (id:40704)") &
    #     (df_kb_r['dataset'] != "credit-g (id:31)") &
    #     (df_kb_r['dataset'] != "breast-cancer-dropped-missing-attributes-values (id:23499)") &
    #     (df_kb_r['dataset'] != "ilpd (id:1480)") &
    #     (df_kb_r['dataset'] != "ilpd-numeric (id:41945)")
    # ]
    
    # # df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",", index=False)
    # print(df_kb_r)
    
    
    # #kb_full_results
    # df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",")
    # df_kb_r = df_kb_r.loc[
    #     #low metrics
    #     (df_kb_r['dataset'] != "pc2 (id:1069)") &
    #     (df_kb_r['dataset'] != "thoracic-surgery (id:1506)") &
    #     (df_kb_r['dataset'] != "PieChart2 (id:1452)") &
    #     (df_kb_r['dataset'] != "ibm-employee-performance (id:43905)") &
        
    #     #repeated datasets
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43893)") &
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43894)") &
    #     (df_kb_r['dataset'] != "climate-model-simulation-crashes (id:1467)") &
    #     (df_kb_r['dataset'] != "MegaWatt1 (id:1442)") &
        
    #     #low metrics after
    #     (df_kb_r['dataset'] != "kc3 (id:1065)") &
    #     (df_kb_r['dataset'] != "kc1 (id:1067)") &
    #     (df_kb_r['dataset'] != "pc1 (id:1068)") &
    #     (df_kb_r['dataset'] != "PizzaCutter3 (id:1444)") &
    #     (df_kb_r['dataset'] != "pc3 (id:1050)") &
    #     (df_kb_r['dataset'] != "CostaMadre1 (id:1446)") &
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43896)") &
    #     (df_kb_r['dataset'] != "lungcancer_GSE31210 (id:1412)") &
    #     (df_kb_r['dataset'] != "PieChart3 (id:1453)") &
    #     (df_kb_r['dataset'] != "MeanWhile1 (id:1449)") &
    #     (df_kb_r['dataset'] != "PizzaCutter1 (id:1443)") &
    #     (df_kb_r['dataset'] != "mw1 (id:1071)") &
    #     (df_kb_r['dataset'] != "yeast_ml8 (id:316)") &
    #     (df_kb_r['dataset'] != "Speech (id:40910)") &
        
    #     #low metrics after part 2
    #     (df_kb_r['dataset'] != "haberman (id:43)") &
    #     (df_kb_r['dataset'] != "autoUniv-au1-1000 (id:1547)") &
    #     (df_kb_r['dataset'] != "balloon (id:914)") &
    #     (df_kb_r['dataset'] != "blood-transfusion-service-center (id:1464)") &
    #     (df_kb_r['dataset'] != "SPECT (id:336)") &
    #     (df_kb_r['dataset'] != "SPECTF (id:1600)") &
    #     (df_kb_r['dataset'] != "kc2 (id:1063)") &
    #     (df_kb_r['dataset'] != "analcatdata_dmft (id:1014)") &
    #     (df_kb_r['dataset'] != "solar-flare (id:40702)") &
    #     (df_kb_r['dataset'] != "mc1 (id:1056)") &
        
    #     #low metrics after part 3
    #     (df_kb_r['dataset'] != "Titanic (id:40704)") &
    #     (df_kb_r['dataset'] != "credit-g (id:31)") &
    #     (df_kb_r['dataset'] != "breast-cancer-dropped-missing-attributes-values (id:23499)") &
    #     (df_kb_r['dataset'] != "ilpd (id:1480)") &
    #     (df_kb_r['dataset'] != "ilpd-numeric (id:41945)")
    # ]
    # # df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_full_results.csv", sep=",", index=False)
    # print(df_kb_r)
    
    
    # #kb_characteristics
    # df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",")
    
    # df_kb_r = df_kb_r.loc[
    #     #low metrics
    #     (df_kb_r['dataset'] != "pc2 (id:1069)") &
    #     (df_kb_r['dataset'] != "thoracic-surgery (id:1506)") &
    #     (df_kb_r['dataset'] != "PieChart2 (id:1452)") &
    #     (df_kb_r['dataset'] != "ibm-employee-performance (id:43905)") &
        
    #     #repeated datasets
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43893)") &
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43894)") &
    #     (df_kb_r['dataset'] != "climate-model-simulation-crashes (id:1467)") &
    #     (df_kb_r['dataset'] != "MegaWatt1 (id:1442)") &
        
    #     #low metrics after
    #     (df_kb_r['dataset'] != "kc3 (id:1065)") &
    #     (df_kb_r['dataset'] != "kc1 (id:1067)") &
    #     (df_kb_r['dataset'] != "pc1 (id:1068)") &
    #     (df_kb_r['dataset'] != "PizzaCutter3 (id:1444)") &
    #     (df_kb_r['dataset'] != "pc3 (id:1050)") &
    #     (df_kb_r['dataset'] != "CostaMadre1 (id:1446)") &
    #     (df_kb_r['dataset'] != "ibm-employee-attrition (id:43896)") &
    #     (df_kb_r['dataset'] != "lungcancer_GSE31210 (id:1412)") &
    #     (df_kb_r['dataset'] != "PieChart3 (id:1453)") &
    #     (df_kb_r['dataset'] != "MeanWhile1 (id:1449)") &
    #     (df_kb_r['dataset'] != "PizzaCutter1 (id:1443)") &
    #     (df_kb_r['dataset'] != "mw1 (id:1071)") &
    #     (df_kb_r['dataset'] != "yeast_ml8 (id:316)") &
    #     (df_kb_r['dataset'] != "Speech (id:40910)") &
        
    #     #low metrics after part 2
    #     (df_kb_r['dataset'] != "haberman (id:43)") &
    #     (df_kb_r['dataset'] != "autoUniv-au1-1000 (id:1547)") &
    #     (df_kb_r['dataset'] != "balloon (id:914)") &
    #     (df_kb_r['dataset'] != "blood-transfusion-service-center (id:1464)") &
    #     (df_kb_r['dataset'] != "SPECT (id:336)") &
    #     (df_kb_r['dataset'] != "SPECTF (id:1600)") &
    #     (df_kb_r['dataset'] != "kc2 (id:1063)") &
    #     (df_kb_r['dataset'] != "analcatdata_dmft (id:1014)") &
    #     (df_kb_r['dataset'] != "solar-flare (id:40702)") &
    #     (df_kb_r['dataset'] != "mc1 (id:1056)") &
        
    #     #low metrics after part 3
    #     (df_kb_r['dataset'] != "Titanic (id:40704)") &
    #     (df_kb_r['dataset'] != "credit-g (id:31)") &
    #     (df_kb_r['dataset'] != "breast-cancer-dropped-missing-attributes-values (id:23499)") &
    #     (df_kb_r['dataset'] != "ilpd (id:1480)") &
    #     (df_kb_r['dataset'] != "ilpd-numeric (id:41945)")
    # ]
    
    # # df_kb_r.to_csv(sys.path[0] + "/output/" + "kb_characteristics.csv", sep=",", index=False)
    # print(df_kb_r)
    
    print("end")

def remove_by_bad_metrics():
    #kb_results
    df_kb_r = pd.read_csv(sys.path[0] + "/output/" + "kb_results.csv", sep=",")
    
    df_kb_r2 = df_kb_r.drop(df_kb_r[
        ~ (
        (df_kb_r['balanced accuracy'] < 0.5) |
        (df_kb_r['f1 score'] < 0.5) |
        (df_kb_r['roc auc'] < 0.5) |
        (df_kb_r['geometric mean'] < 0.5) |
        (df_kb_r['cohen kappa'] < 0.5)
        )
        ].index)
    
    print(df_kb_r2)
    # TAKE NOTES TO REMOVE

    print("end")


# remove_all_worst_datasets()

# remove_by_bad_metrics()

