from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from src.metadata import D_REPLACE_METHOD_NAMES


def keep_equalsize_min100samples_datasets(df_method_alldatasets):
    """
    Filter on compatible equal-size data sets with at least 100 samples.
    """

    cwd = Path.cwd()
    df_ucr_data_summary = pd.read_csv(cwd / "data/DataSummary_prep_equalsize_min100samples.csv")
    l_datasets_ucr = df_ucr_data_summary["Name"].unique().tolist()

    df_method_alldatasets = df_method_alldatasets.query(
        f"dataset in {l_datasets_ucr}"
    )
    return df_method_alldatasets


def drop_features(df_acc_method_alldatasets):
    """Drop some features.
    """

    l_feat_drop = [
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "params",
        "rank_test_score"
    ]

    for col in df_acc_method_alldatasets.columns:
        if col.startswith("split"):
            l_feat_drop.append(col)

    df_acc_method_alldatasets.drop(columns=l_feat_drop, inplace=True)

    return df_acc_method_alldatasets


def rename_feature_names(df_acc_method_alldatasets):
    """Rename the parameter grid features without the name of the pipeline
    step.
    """

    d_rename_features = dict()
    for col in df_acc_method_alldatasets.columns.tolist():
        if col.startswith("param_smoother"):
            d_rename_features[col] = "smoother_"+col.split("__")[1]
        elif col.startswith("param_"):
            d_rename_features[col] = col.split("__")[1]
    df_acc_method_alldatasets.rename(columns=d_rename_features, inplace=True)
    if "alphabet_size_slope" in df_acc_method_alldatasets.columns:
        # for 1d-SAX tslearn
        d_rename_features["alphabet_size_avg"] = "n_symbols_mean"
        d_rename_features["alphabet_size_slope"] = "n_symbols_slope"
    elif "alphabet_size_avg" in df_acc_method_alldatasets.columns: 
        # for SAX tslearn
        d_rename_features["alphabet_size_avg"] = "n_symbols"
    df_acc_method_alldatasets.rename(columns=d_rename_features, inplace=True)
    return df_acc_method_alldatasets


def add_nsamples(df_method_alldatasets):
    """
    Add the `n_samples` feature (deleting existing one if needed).
    """

    # Delete `n_samples` if it already exists
    col = "n_samples"
    if col in df_method_alldatasets.columns.tolist():
        df_method_alldatasets.drop(columns=[col], inplace=True)

    # Getting `n_samples` per data sets
    cwd = Path.cwd()
    df_ucr_data_summary = pd.read_csv(cwd / "data/DataSummary_prep_equalsize_min100samples.csv")
    df_nsamples_alldatasets = (
        df_ucr_data_summary.rename(
            columns={
                "Name": "dataset",
                "Length": "n_samples",
            }
        )[["dataset", "n_samples"]]
        .drop_duplicates()
        .sort_values(by=["dataset"])
        .reset_index(drop=True)
    )

    # Add `n_samples` per data sets
    df_method_alldatasets = df_method_alldatasets.merge(
        df_nsamples_alldatasets)

    return df_method_alldatasets


def get_param_columns_float(df):
    """
    While cleaning the DataFrame of accuracies after a grid search for a
        symbolic representation, get the columns with only nan values who have
        to be transformed into None (in a separate function).
    """

    # Get columns that are parameters (float or not)
    l_param_columns = [
        col for col in df.columns.tolist() if col.startswith("param_")
    ]

    # Get columns that are parameters and of float type
    l_param_columns_float = (
        df[l_param_columns]
        .select_dtypes(include=["float16", "float32", "float64"])
        .columns.tolist()
    )

    b_get_param_columns_float = Bunch(
        l_param_columns=l_param_columns,
        l_param_columns_float=l_param_columns_float,
    )
    return b_get_param_columns_float


def nan2none(df, bool_all=False):
    """
    Transform columns with only NaN values into None values (so that the None
        values are compatible with the assert tests of the package).
    """

    df_output = df.copy()

    # Get columns that are parameters of float type
    if not(bool_all):
        b_get_param_columns_float = get_param_columns_float(df)
        l_param_columns_float = b_get_param_columns_float.l_param_columns_float
    else:
        l_param_columns_float = df_output.columns.tolist()

    # Transform the features that are parameters of float type with only `NaN`
    # into `None`
    len_df = len(df_output)
    for column in l_param_columns_float:
        if df_output[column].isna().sum() == len_df:
            print(column)
            df_output[column] = None

    return df_output


def clean_acc_method_alldatasets(
    df_acc_method_alldatasets: pd.DataFrame,
    method_abbreviate: str,
    is_verbose=False,
):
    """Cleaning the data frames of accuracies before computing the memory usage.

    For symbolic methods with uniform segmentation.

    Cleaning tasks done:
        - Drop some features
        - Rename the features
        - Fill `n_samples` if needed
        - Sort rows
        - Filter on the current method
        - Display missing values per feature
        - Convert NaN values into None when applicable
        - Delete non equal-sized data sets
        - Display number of data sets
        - `n_segments` can not be superior to `n_samples`
        - Check that the test score is valid
        - Insert method name
    """

    df_acc_method_alldatasets = drop_features(df_acc_method_alldatasets)
    method_name = D_REPLACE_METHOD_NAMES[method_abbreviate]

    df_acc_method_alldatasets = rename_feature_names(df_acc_method_alldatasets)

    # Fill `n_samples` if needed
    if "n_samples" not in df_acc_method_alldatasets.columns or (
            df_acc_method_alldatasets["n_samples"].isna().sum() == len(df_acc_method_alldatasets)):
        df_acc_method_alldatasets = add_nsamples(df_acc_method_alldatasets)

    # Sort rows
    df_acc_method_alldatasets.sort_values(
        by=df_acc_method_alldatasets.columns.tolist(),
        inplace=True
    )

    # Keep only the equal-size data sets with at least 100 samples
    df_acc_method_alldatasets = keep_equalsize_min100samples_datasets(
        df_acc_method_alldatasets)

    # Display missing values and replace by `None`
    if is_verbose:
        print("Percentage of missing values per feature (will be transformed to `None`):")
    df_nan_features = (
        df_acc_method_alldatasets.isna().sum() / len(df_acc_method_alldatasets) * 100
    ).round(0).reset_index().rename(columns={"index": "feature", 0: "perc_nan"}).query("perc_nan > 0")
    if is_verbose:
        print(df_nan_features)
    l_fullnan_features = df_nan_features.query("perc_nan == 100")[
        "feature"].tolist()
    for col in l_fullnan_features:
        df_acc_method_alldatasets[col] = None

    # Display number of data sets
    n_datasets = df_acc_method_alldatasets["dataset"].nunique()
    if is_verbose:
        print(
            f"\nNumber of unique datasets for {method_name}"
            f":\n\t{n_datasets}"
        )

    # `n_segments` can not be superior to `n_samples`
    if not("n_segments" in df_nan_features["feature"].tolist()):
        perc = np.round(
            len(df_acc_method_alldatasets.query("n_segments > n_samples"))
            / len(df_acc_method_alldatasets) * 100, 2
        )
        if is_verbose:
            print(
                "\nPercentage of lines with n_segments > n_samples"
                f" that have been deleted:\n\t{perc}%"
            )
        df_acc_method_alldatasets = df_acc_method_alldatasets.query(
            "n_segments <= n_samples"
        )
    else:
        print("\nNo filtering done on n_segments <= n_samples")

    # Param grid with no fit
    perc_nofit = round(
        len(df_acc_method_alldatasets.query("mean_test_score == 0"))
        / len(df_acc_method_alldatasets) * 100, 2
    )
    if is_verbose:
        print(
            "\nPercentage of param grid with no fit (null test score)"
            f" that have been deleted:\n\t{perc_nofit}%"
        )
    df_acc_method_alldatasets = df_acc_method_alldatasets.query(
        "mean_test_score > 0")

    # Add method name
    df_acc_method_alldatasets.insert(0, "method", method_name)

    # For 1d-SAX, compute `n_symbols`
    if method_name == "1d-SAX":
        df_acc_method_alldatasets["n_symbols"] = df_acc_method_alldatasets["n_symbols_mean"] * \
            df_acc_method_alldatasets["n_symbols_slope"]

    return df_acc_method_alldatasets, l_fullnan_features