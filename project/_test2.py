import pandas as pd
import openml.datasets


def read_file_openml(id):
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe")

    df = pd.DataFrame(X, columns=attribute_names)
    df["class"] = y
    
    openml.datasets.get_dataset(id, download_data=False)
    
    dataset_name = dataset.name
    
    return df, dataset_name

df, dataset_name = read_file_openml(450)

print(df)
print(dataset_name)
