import sys
from decimal import Decimal
import pandas as pd
import openml.datasets
from ml import read_file_openml

def step_1():

    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    # print(datalist)
    #4124 datasets

    datalist = datalist.query("status == 'active' & NumberOfClasses == 2 & NumberOfInstances > 200 & NumberOfInstances < 10000 & NumberOfFeatures < 500 & NumberOfInstancesWithMissingValues == 0")

    # print(datalist)

    #problem at line 210 - id:40588 ...
    # print(datalist.iloc[210,:])

    # 337 datasets -> 267 datasets

    i=0
    for id in datalist["did"]:
        print(i)
        i +=1
        
        try:
            dataset = openml.datasets.get_dataset(id)

            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute, dataset_format="dataframe")

            df = pd.DataFrame(X, columns=attribute_names)
            df["class"] = y

            #X = df.iloc[: , :-1]
            y = df.iloc[: , -1:]

            #print(y["class"].dtype.name)

            encoded_columns = []
            for column_name in y.columns:
                if y[column_name].dtype == object or y[column_name].dtype.name == 'category':
                    encoded_columns.extend([column_name])
                else:
                    pass

            if encoded_columns:
                y = pd.get_dummies(y, y[encoded_columns].columns, drop_first=True)


            imbalance_ratio = 0
            if y.values.tolist().count([0]) > 0 and y.values.tolist().count([1]) > 0:
                if y.values.tolist().count([0]) >= y.values.tolist().count([1]):
                    imbalance_ratio = round(y.values.tolist().count([0])/y.values.tolist().count([1]),3)
                else:
                    imbalance_ratio = round(y.values.tolist().count([1])/y.values.tolist().count([0]),3)

            # print("imbalance ratio      :", imbalance_ratio)
            # print("")
            
            
            df_openml = pd.read_csv(sys.path[0] + "/input/" + "datasets_openml.csv", sep=",")
            
            df_openml.loc[len(df_openml.index)] = [Decimal(id), imbalance_ratio]
            
            df_openml.to_csv(sys.path[0] + "/input/" + "datasets_openml.csv", sep=",", index=False)
        
        except:
            print("An exception occurred with id", id)

def step_2():
    df_openml = pd.read_csv(sys.path[0] + "/input/" + "datasets_openml.csv", sep=",")

    rows_to_keep = df_openml["imbalance ratio"] > 5

    # imbalance ratio > 10 -> 22 datasets
    # imbalance ratio > 5  -> 63 datasets

    df_openml = df_openml[rows_to_keep]

    df_openml.sort_values(by=['imbalance ratio'], inplace=True)

    pd.set_option('display.max_rows', df_openml.shape[0]+1)
    # print(df_openml)
    # #976 ... 1069
    
    # print("\ncount:")
    # print(df_openml.count())
    
    #get 4 random datasets whithout 1069 and 1056
    df_openml = df_openml.loc[(df_openml["id"] != 1069) & (df_openml["id"] != 1056)]
    print(df_openml.sample(n=4))
    
def step_3():
    df, dataset_name = read_file_openml(40910)

    print("\ndf:")
    print(df)

    print("\ndataset_name:")
    print(dataset_name)


# step_1()
step_2()
# step_3()
