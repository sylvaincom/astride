
from pathlib import Path

import pandas as pd
from src.utils_reconstruction import launch_reconstruction_error_denom_dataset
from src.utils import concatenate_df

def main(
    denom,
    n_symbols,
    abba_scl,
    date_exp,
):

    # Set the path
    cwd = Path.cwd()
    path = Path(cwd / "results" / date_exp / "reconstruction" / str(denom))

    # Consider univariate and equal-size data sets with at least 100 samples
    df_datasets_total = pd.read_csv(
        cwd / "data/DataSummary_prep_equalsize_min100samples.csv")
    l_datasets_total = df_datasets_total["Name"].unique().tolist()

    # Remove the data sets with computing issues
    l_datasets_problems = [
        "DodgerLoopWeekend",
        "DodgerLoopGame",
        "DodgerLoopDay",
        "MelbournePedestrian",
        "UWaveGestureLibraryZ",
    ]
    l_datasets_scope = [
        dataset for dataset in l_datasets_total if dataset not in l_datasets_problems]
    print("Total number of data sets in the scope:", len(l_datasets_scope))

    # Ignore the data sets that have already been computed (to not compute again)
    l_csvfiles_computed_datasets = list(
        path.rglob(f"reconstruction_errors_*.csv"))
    if len(l_csvfiles_computed_datasets) > 0:
        df_computed_datasets = concatenate_df(
            l_csvfiles_computed_datasets).drop_duplicates()
        l_datasets_computed = df_computed_datasets["dataset"].unique().tolist()
        print("Number of data sets already computed:", len(l_datasets_computed))

        l_datasets_to_compute = [
            dataset for dataset in l_datasets_scope if dataset not in l_datasets_computed]
        print("Number of new data sets to compute:", len(l_datasets_to_compute))
    else:
        l_datasets_to_compute = l_datasets_scope
        print("Number of data sets to compute:", len(l_datasets_to_compute))

    # Launch the signal reconstruction task, for all methods

    print(f"\n\n====\n{date_exp = }\n{denom = }\n=====")

    for (i, dataset_name_ucr) in enumerate(l_datasets_to_compute):
        print(f"Dataset: {dataset_name_ucr}: {i+1}/{len(l_datasets_to_compute)}.")

        try:
            launch_reconstruction_error_denom_dataset(
                denom=denom,
                dataset_name_ucr=dataset_name_ucr,
                n_symbols=n_symbols,
                abba_scl=abba_scl,
                date_exp=date_exp,
            )
        except:
            print(f"--ERROR: {dataset_name_ucr} did not pass!")
            pass

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--denom",
        type=int,
        help="denom is the inverse of the target memory usage ratio, in {3, 4, 5, 6, 10, 15, 20}.",
        required=True,
    )
    parser.add_argument(
        "--date_exp",
        type=str,
        help="Date of the launch of the experiments (for versioning).",
        required=True,
    )
    parser.add_argument(
        "--n_symbols",
        type=int,
        help="Fixed alphabet size for all methods (set to 9 by default).",
        required=False,
        default=9,
    )
    parser.add_argument(
        "--abba_scl",
        type=float,
        help="Fixed scaling parameter for ABBA (set to 1 by default).",
        required=False,
        default=1,
    )

    args = parser.parse_args()
    main(
        denom=args.denom,
        date_exp=args.date_exp,
        n_symbols=args.n_symbols,
        abba_scl=args.abba_scl,
        )
