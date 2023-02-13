import numpy as np
import pandas as pd
from loadmydata.load_uea_ucr import load_uea_ucr_data
from sklearn.utils import Bunch


def load_ucr_dataset(dataset_ucr_name):
    """
    Load a specific UCR data set using the `loadmydata` Python package.

    We do not z-normalize the data here.
    """

    data_ucr = load_uea_ucr_data(dataset_ucr_name)

    X_train = data_ucr.X_train.data[:, :, 0]
    y_train = data_ucr.y_train
    X_test = data_ucr.X_test.data[:, :, 0]
    y_test = data_ucr.y_test

    X_train_test = np.concatenate((X_train, X_test), axis=0)
    y_train_test = np.concatenate((y_train, y_test), axis=0)

    l_train = [X_train[i] for i in np.arange(len(X_train))]
    l_test = [X_test[i] for i in np.arange(len(X_test))]
    l_train_test = [X_train_test[i] for i in np.arange(len(X_train_test))]

    b_load_ucr_data = Bunch(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_test=X_train_test,
        y_train_test=y_train_test,
        l_train=l_train,
        l_test=l_test,
        l_train_test=l_train_test,
    )
    return b_load_ucr_data


def create_path(path):
    """Create path to be able to export files, figures, etc.
    """

    # Get the complete list of parent paths (including the wanted final path)
    list_parent_paths = [path] + list(path.parents)

    # Get the list of paths that do not exist, thus need to be created
    parent_path_to_create = []
    for path in list_parent_paths:
        if not(path.exists()):
            parent_path_to_create.append(path)
    parent_path_to_create = parent_path_to_create[::-1]

    # Create the (needed) paths
    for path in parent_path_to_create:
        if not(path.exists()):
            path.mkdir(exist_ok=True)


def concatenate_df(l_csv_files: list):
    """
    Concatenate a list of csv files into a big pandas DataFrame.
    """
    l_csv_files.sort()
    l_df = []
    for csv_file in l_csv_files:
        df = pd.read_csv(csv_file)
        l_df.append(df)
    df = pd.concat(l_df, ignore_index=True)
    return df
