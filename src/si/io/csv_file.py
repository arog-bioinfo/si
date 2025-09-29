import numpy as np
from si.data.dataset import Dataset
import pandas as pd

def read_csv(filename: str, sep: str, features: bool, label: bool) -> Dataset:
    
    dataframe = pd.read_csv(filepath_or_buffer=filename, sep=sep)

    if features and label:
        X = dataframe.iloc[:, :-1].to_numpy()
        y = dataframe.iloc[:, -1].to_numpy()
        feature_names = dataframe.columns[:-1]
        label_name = dataframe.columns[-1]
        return Dataset(X=X, y=y, features=feature_names, label=label_name)
    
    elif features:
        X = dataframe.to_numpy()
        feature_names = dataframe.columns
        return Dataset(X=X, features=feature_names)
    
    elif label:
        X = np.array()
        y = dataframe.iloc[:, -1].to_numpy()
        label_name = dataframe.columns[-1]
        return Dataset(X = X, y=y, label=label_name)
    else:
        return None
    
def write_csv(filename: str,
              dataset: Dataset,
              sep: str = ',',
              features: bool = False,
              label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file

    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False
    """

    df = pd.DataFrame(dataset.X)
    if features:
        df.columns = dataset.features

    if label:
        y = dataset.y
        label_name = dataset.label
        df[label_name] = y

    else:
        y = None
        label_name = None

    df.to_csv(filename, sep=sep, index=False)

    

