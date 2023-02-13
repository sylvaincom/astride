import ast
import pprint
from datetime import timedelta
from pathlib import Path
from time import time

import pandas as pd
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     PredefinedSplit)

from src.utils import load_ucr_dataset, create_path

pp = pprint.PrettyPrinter()


def launch_grid_search_dataset(pipe, param_grid, b_load_ucr_dataset):
    """
    On a specific data set, launch a 1-Nearest Neighbord classification
        using the default single train / test split (UCR archive)
        on a method with a parameter grid.
    """

    # Retrieve the data
    X_train = b_load_ucr_dataset.X_train
    X_train_test = b_load_ucr_dataset.X_train_test
    y_train_test = b_load_ucr_dataset.y_train_test

    # Define the test fold
    train_size = len(X_train)
    train_test_size = len(X_train_test)
    test_fold = [-1]*train_size + [1]*(train_test_size-train_size)
    cv_input = PredefinedSplit(test_fold)

    # Launch grid search
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        n_jobs=-1,
        cv=cv_input,
        verbose=1,
        error_score=0.0,  # np.nan
    )
    grid_search.fit(X_train_test, y_train_test)

    print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    print(grid_search.best_params_)

    return grid_search


def launch_grid_search_acc_datasets(
    l_datasets_bench: list,
    method_name: str,
    pipe,
    param_grid: list or dict,
    date_exp="unknown",
):
    """
    On all data sets in `l_datasets_bench`, launch a 1-Nearest Neighbord
        classification using the default UCR archive single train / test split.

    Once a data set is classified, the DataFrame with the results from the grid
        search is exported.

    The `method_name` must coincide with the `param_grid` (not checked in
    this function).
    """

    # List of DataFrames containing the results of the grid search on the datasets
    l_df_acc_method_alldatasets = list()

    # Iterate over the datasets
    for i, dataset_name_ucr in enumerate(l_datasets_bench):

        print(
            f"\n\n===== Data set: {dataset_name_ucr} ({i+1}/{len(l_datasets_bench)}) ======"
        )

        # Start measuring the time
        start_time = time()

        param_grid_dataset = param_grid.copy()

        # Load the data
        b_load_ucr_dataset = load_ucr_dataset(dataset_name_ucr)

        # Set the number of samples (when needed)
        if "symbolization__lookup_table_type" in param_grid_dataset.keys():
            if param_grid_dataset["symbolization__lookup_table_type"][0] == "mindist" or param_grid_dataset["symbolicsignaldistance__distance"][0] == "euclidean":
                n_samples = len(b_load_ucr_dataset.l_train[0])
                param_grid_dataset["symbolicsignaldistance__n_samples"] = [
                    n_samples]

        # Discard the param settings that have already been launched
        cwd = Path.cwd()
        folder = cwd / "results" / date_exp / "acc"
        dir_path = Path(
            folder / f"df_acc_{method_name}_{dataset_name_ucr}.csv")
        if dir_path.is_file():

            # Get the list of parameter settings in our scope
            l_param_settings_scope = list(ParameterGrid(param_grid_dataset))

            # Get the list of parameter settings that have already been computed
            df_acc_method_dataset_exist = pd.read_csv(dir_path)
            l_param_settings_computed = [ast.literal_eval(
                param_setting) for param_setting in df_acc_method_dataset_exist["params"].tolist()]

            # Get the list of parameter settings left to compute
            l_param_settings_to_compute = list()
            l_param_settings_to_compute = [
                param_setting for param_setting in l_param_settings_scope
                if param_setting not in l_param_settings_computed
            ]

            if len(l_param_settings_to_compute) == 0:
                print(f"No param settings left to compute for {dataset_name_ucr}!")
                continue
            else:
                # Transform the values of `param_setting` into a list of one element
                for (i, param_setting) in enumerate(l_param_settings_to_compute):
                    for key in param_setting.keys():
                        param_setting[key] = [param_setting[key]]
                    l_param_settings_to_compute[i] = param_setting

            # Display
            print(
                f"\nTotal number of parameter settings in the scope: \n\t{len(l_param_settings_scope)}")
            print(
                f"Number of parameter settings already computed: \n\t{len(l_param_settings_computed)}")
            print(
                f"Number of new parameter settings left to compute: \n\t{len(l_param_settings_to_compute)}")
            print(
                f"New parameter settings left to compute: \n\t{l_param_settings_to_compute[0]}")

        else:
            l_param_settings_to_compute = param_grid_dataset.copy()

        # Launch the grid search on the data set
        grid_search = launch_grid_search_dataset(
            pipe=pipe, param_grid=l_param_settings_to_compute, b_load_ucr_dataset=b_load_ucr_dataset,
        )

        # Prepare the output data
        df_acc_method_dataset_computed = pd.DataFrame(grid_search.cv_results_)
        df_acc_method_dataset_computed.insert(
            loc=0, column='dataset', value=dataset_name_ucr)

        # Create folder if needed (for the export)
        cwd = Path.cwd()
        folder = Path(cwd / "results" / date_exp / "acc")
        create_path(folder)

        # If needed, concatenate the new results with the already computed ones
        if dir_path.is_file():
            df_acc_method_dataset = pd.concat(
                [df_acc_method_dataset_exist, df_acc_method_dataset_computed], ignore_index=True)
            #TODO: maybe sort the obtained data frame
        else:
            df_acc_method_dataset = df_acc_method_dataset_computed.copy()

        # Export the data
        df_acc_method_dataset.to_csv(
            folder / f"df_acc_{method_name}_{dataset_name_ucr}.csv",
            index=False,
        )
        l_df_acc_method_alldatasets.append(df_acc_method_dataset)

        # Stop measuring the time and printing it
        stop_time = time()
        dataset_time = timedelta(seconds=stop_time - start_time)
        print(f"\nTime for this data set:\n\t{dataset_time}")

    if len(l_df_acc_method_alldatasets)==0:
        print(f"\n=====\nFor all data sets, no new param setting were computed!")
        return None
    else:
        df_acc_method_alldatasets = pd.concat(
            l_df_acc_method_alldatasets, ignore_index=True
        )
        return df_acc_method_alldatasets