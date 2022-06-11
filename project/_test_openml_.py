import sys
from decimal import Decimal
import pandas as pd
import openml.datasets

openml_list = openml.datasets.list_datasets()  # returns a dict
datalist = pd.DataFrame.from_dict(openml_list, orient="index")
#print(datalist)

datalist = datalist.query("status == 'active' & NumberOfClasses == 2 & NumberOfInstances > 200 & NumberOfInstances < 1000 & NumberOfFeatures < 50 & NumberOfInstancesWithMissingValues == 0")
#print(datalist)
#122 datasets


i=0
for id in datalist["did"]:
    print(i)
    i +=1
    
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


