from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import euclidean
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch
from tslearn.metrics import dtw
from tslearn.piecewise import (OneD_SymbolicAggregateApproximation,
                               SymbolicAggregateApproximation)
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from ABBA.ABBA import ABBA
from src.segment_feature import SegmentFeature
from src.segmentation import Segmentation
from src.symbolization import Symbolization
from src.utils import concatenate_df, load_ucr_dataset, create_path
from src.utils_sfa import test_SFA, train_SFA

plt.rcParams['figure.figsize'] = (10, 3)
sns.set_theme()
cmap = LinearSegmentedColormap.from_list('gr', ["g", "w", "r"], N=256)
cwd = Path.cwd()


def ts_to_list(X):
    """Format the signals into lists in a dictionary.
    ABBA code inputs signals as lists.
    No scaling is applied here.
    """
    d_ts_list = dict()
    for signal_index in range(len(X)):
        X_signal = X[signal_index]
        ts_list = [elem[0] for elem in X_signal]
        d_ts_list[signal_index] = ts_list
    return d_ts_list


def get_nsegments_abba(dataset_name_ucr, X, n_symbols, abba_scl, denom):
    """Launch the compression of ABBA signal by signal (accross the whole
    data set).
    No scaling is applied here.
    """

    d_ts_list = ts_to_list(X)
    n_samples = len(d_ts_list[0])

    d_tol_used = dict()
    for signal_index in range(len(X)):
        # ABBA: adjust tolerance, for example so at least 20% compression is
        # used if denom=5
        ts_list = d_ts_list[signal_index]
        abba_tol = 0.05
        abba = ABBA(tol=abba_tol, min_k=n_symbols,
                    max_k=n_symbols, scl=abba_scl, verbose=0)
        pieces = abba.compress(ts_list)
        nsegments_abba = len(pieces)
        while nsegments_abba > n_samples/denom:
            abba_tol += 0.05
            abba = ABBA(tol=abba_tol, min_k=n_symbols,
                        max_k=n_symbols, scl=abba_scl, verbose=0)
            pieces = abba.compress(ts_list)
            nsegments_abba = len(pieces)
        d_tol_used[signal_index] = [abba_tol, nsegments_abba]

        # Check the number of segments
        err_msg = "There are more symbols (clusters) than segments."
        assert nsegments_abba >= n_symbols, err_msg

    # Get the mean number of segments for all ABBA symbolic sequences
    df_tol_abba = (
        pd.DataFrame.from_dict(d_tol_used, orient="index").reset_index()
        .rename(columns={"index": "signal_index", 0: "abba_tol", 1: "nsegments_abba"})
    )
    err_msg = "The protocol for setting the number of segments is flawed."
    assert len(df_tol_abba) == len(X), err_msg
    df_tol_abba.insert(0, "denom", denom)
    df_tol_abba.insert(1, "dataset", dataset_name_ucr)
    df_tol_abba.insert(4, "n_symbols", n_symbols)
    n_segments_mean = round(df_tol_abba.nsegments_abba.mean())
    df_tol_abba.insert(6, "n_segments", n_segments_mean)

    return df_tol_abba


def transform_invtransform_abba(X, df_tol_abba, n_symbols, abba_scl):
    """
    Transform and inverse transform a data set of signals using ABBA.
    """
    d_ts_list = ts_to_list(X)
    l_symb_ts_abba_inv = list()
    for signal_index in range(len(X)):
        ts_list = d_ts_list[signal_index]
        abba_tol = df_tol_abba.query(f"signal_index == {signal_index}")[
            "abba_tol"].values[0]
        abba = ABBA(tol=abba_tol, min_k=n_symbols,
                    max_k=n_symbols, scl=abba_scl, verbose=0)
        symb_ts_abba, centers = abba.transform(ts_list)
        symb_ts_abba_inv = abba.inverse_transform(
            symb_ts_abba, centers, ts_list[0])
        l_symb_ts_abba_inv.append(symb_ts_abba_inv)
    arr_symb_ts_abba_inv = np.array(l_symb_ts_abba_inv)
    return arr_symb_ts_abba_inv


def transform_symb_ts(pipe, X):
    """Symbolization of a data set of signals.
    """
    Xt = X
    for name, transform in pipe.steps:
        if name == "kneighborsclassifier":
            break
        if transform is not None:
            Xt = transform.transform(Xt)
        if name=="segmentation":
            b_transform_segmentation = Xt
            list_of_bkps = b_transform_segmentation.list_of_bkps
            list_of_scaled_signals = b_transform_segmentation.list_of_signals
        if name=="symbolization":
            b_transform_symbolization = Xt
            list_of_symbolic_signals = b_transform_symbolization.list_of_symbolic_signals
            lookup_table = b_transform_symbolization.lookup_table
            features_with_symbols_nonumreduc_noquantifseglen_df = b_transform_symbolization._features_with_symbols_nonumreduc_noquantifseglen_df
            features_with_symbols_noquantifseglen_df = b_transform_symbolization._features_with_symbols_noquantifseglen_df
            features_with_symbols_df = b_transform_symbolization._features_with_symbols_df
            y_quantif_bins = pipe["symbolization"].y_quantif_bins_
    b_transform_symb_ts = Bunch(
        list_of_bkps=list_of_bkps,
        list_of_scaled_signals=list_of_scaled_signals,
        list_of_symbolic_signals=list_of_symbolic_signals,
        lookup_table=lookup_table,
        features_with_symbols_nonumreduc_noquantifseglen_df=features_with_symbols_nonumreduc_noquantifseglen_df,
        features_with_symbols_noquantifseglen_df=features_with_symbols_noquantifseglen_df,
        features_with_symbols_df=features_with_symbols_df,
        y_quantif_bins=y_quantif_bins,
    )
    return b_transform_symb_ts


def inv_transform_symb_ts(
    features_with_symbols_df,
):
    """Inverse transformation of a data set of symbolic sequences.
    The feature per segment must be the mean.
    Works for ASTRIDE and FASTRIDE for example.
    """

    # Map a symbol to its quantized mean
    d_map_symbol_mean = (
        features_with_symbols_df
        [["mean_feat", "segment_symbol"]]
        .groupby(by=["segment_symbol"])
        .mean()
    ).to_dict()["mean_feat"]

    # Get the list of symbolic signals with replication according to the
    # segment lengths
    list_of_replicated_symbolic_signals = list()
    for (_, group) in features_with_symbols_df.groupby("signal_index"):
        list_of_replicated_symbolic_signals.append(
            np.array(
                group.segment_symbol.apply(lambda x: [x])
                * group.segment_length.astype(int)
            ).sum()
        )

    # Get the inverse transform of the replicated symbolic signals
    list_of_inv_symbolic_signals = list()
    for duplicated_symbolic_signal in list_of_replicated_symbolic_signals:
        inv_symbolic_signal = pd.Series(
            duplicated_symbolic_signal).map(d_map_symbol_mean).tolist()
        list_of_inv_symbolic_signals.append(inv_symbolic_signal)
    arr_symb_ts_method_inv = np.array(list_of_inv_symbolic_signals)

    b_inv_transform_symb_ts = Bunch(
        d_map_symbol_mean=d_map_symbol_mean,
        arr_symb_ts_method_inv=arr_symb_ts_method_inv,
    )
    return b_inv_transform_symb_ts


def get_reconstruction_errors(reconstruction_errors, methods):

    df_reconstruction_errors = (
        pd.DataFrame(reconstruction_errors)
    )
    df_reconstruction_errors.columns = methods
    df_reconstruction_errors = (
        df_reconstruction_errors.reset_index()
        .rename(columns={"index": "signal_index"})
    )

    return df_reconstruction_errors


def truncate_signals(X_signals, n_segments):
    """Truncate all signals in the data set.
    """
    n_samples = len(X_signals[0])
    trunc_index = (n_samples//n_segments)*n_segments
    X_signals_trunc = X_signals[:, :trunc_index]
    return X_signals_trunc


def compute_reconstruction_errors(X_signals, d_reconstructed_signals):
    """For all methods in the keys of `d_reconstructed_signals`, compute the
    recontruction error.
    """

    l_reconstruction_errors = list()
    for method in d_reconstructed_signals:
        X_reconstructed_signals = d_reconstructed_signals[method]
        for signal_index in range(len(X_signals)):
            original_signal = X_signals[signal_index, :, 0]
            reconstructed_signal = X_reconstructed_signals[signal_index, :]
            eucl_error = euclidean(original_signal, reconstructed_signal)
            dtw_error = dtw(original_signal, reconstructed_signal)
            l_reconstruction_errors.append(
                [method, signal_index, eucl_error, dtw_error])
    df_reconstruction_errors = pd.DataFrame(l_reconstruction_errors)
    df_reconstruction_errors.columns = [
        "method", "signal_index", "eucl_error", "dtw_error"]
    return df_reconstruction_errors


def launch_reconstruction_error_denom_dataset(
    denom: int,
    dataset_name_ucr: str,
    n_symbols: int = 9,
    abba_scl=1,
    date_exp: str = "unknown",
    is_compute_error_not_truncated: bool = False,
):
    """Fit on the train set and transform on the train set because it is
    reconstruction.
    Does the transform, the inverse transform, then computes the
        reconstruction errors.
    For a specific data set and denom value (it will find the tolerance and
        the number of segments).
    For 1d-SAX, A=9 corresponds to (A_mean, A_slope) = (3, 3) for example.
        Hence, A must be a squared in {4, 9, 16, 25, 36}.
    Because of 1d-SAX, we need to truncate the end values from the reconstruction.
    The memory usage is w/n (there is notion of compression rate).

    The issue without computing the errors on the not truncation as well is
        the possiblity of nan of inf values from 1d-SAX which leads to a Python
        error in the compute of the Euclidean distance.
    """

    # ==========================================================================
    # Load the data
    b_load_ucr_dataset = load_ucr_dataset(dataset_name_ucr)
    X_train = b_load_ucr_dataset.X_train
    n_samples = len(X_train[0])

    # For 1d-SAX
    err_msg = "`n_symbols` must be in {4, 9, 16, 25, 36}."
    assert n_symbols in [4, 9, 16, 25, 36], err_msg
    n_symbols_avg = round(np.sqrt(n_symbols))
    n_symbols_slope = round(np.sqrt(n_symbols))

    # The reconstruction is performed on the scaled signals
    scaler = TimeSeriesScalerMeanVariance()
    X_train_scaled = scaler.fit_transform(X_train)

    # ==========================================================================
    # Launch ABBA and set the mean number of segments for all other methods

    # Launch the compression of ABBA signal by signal (accross the whole train set)
    # The ABBA compression is applied to the scaled signals
    df_tol_abba = get_nsegments_abba(
        dataset_name_ucr=dataset_name_ucr,
        X=X_train_scaled,
        n_symbols=n_symbols,
        abba_scl=abba_scl,
        denom=denom
    )
    # Retrieve the mean number of segments from ABBA on this data set
    n_segments_mean = df_tol_abba["n_segments"].unique()[0]

    # ==========================================================================
    # Build the methods' pipelines (without scaling again)
    # Set the number of segments for all the remaining methods

    # SAX (without re-scaling the signals)
    sax = SymbolicAggregateApproximation(
        n_segments=n_segments_mean,
        alphabet_size_avg=n_symbols,
        scale=False,
    )

    # 1d-SAX (without re-scaling the signals)
    one_d_sax = OneD_SymbolicAggregateApproximation(
        n_segments=n_segments_mean,
        alphabet_size_avg=n_symbols_avg,
        alphabet_size_slope=n_symbols_slope,
        scale=False,
    )

    # ASTRIDE (without re-scaling the signals in the pipeline)
    astride = (
        make_pipeline(
            Segmentation(
                univariate_or_multivariate="multivariate",
                uniform_or_adaptive="adaptive",
                mean_or_slope="mean",
                n_segments=n_segments_mean,
                pen_factor=None
            ),
            SegmentFeature(
                features_names=["mean"]
            ),
            Symbolization(
                n_symbols=n_symbols,
                symb_method="quantif",
                symb_quantif_method="quantiles",
                symb_cluster_method=None,
                features_scaling=None,
                reconstruct_bool=True,
                n_regime_lengths="divide_exact",
                seglen_bins_method=None,
                lookup_table_type="mof"
            ),
        )
    )

    # FASTRIDE (without re-scaling the signals in the pipeline)
    fastride = (
        make_pipeline(
            Segmentation(
                univariate_or_multivariate="multivariate",
                uniform_or_adaptive="uniform",
                mean_or_slope=None,
                n_segments=n_segments_mean,
                pen_factor=None
            ),
            SegmentFeature(
                features_names=["mean"]
            ),
            Symbolization(
                n_symbols=n_symbols,
                symb_method="quantif",
                symb_quantif_method="quantiles",
                symb_cluster_method=None,
                features_scaling=None,
                reconstruct_bool=False,
                n_regime_lengths=None,
                seglen_bins_method=None,
                lookup_table_type="mof"
            ),
        )
    )

    # ==========================================================================
    # For each method: get the symbolic sequences (transform) and reconstruct
    # (inverse transfrom) them, on the train set only

    # ABBA
    arr_symb_ts_abba_inv = transform_invtransform_abba(
        X=X_train_scaled,
        df_tol_abba=df_tol_abba,
        n_symbols=n_symbols,
        abba_scl=abba_scl,
    )

    # SAX
    arr_symb_ts_sax = sax.fit_transform(X_train_scaled, scale=False)
    arr_symb_ts_sax_inv = sax.inverse_transform(arr_symb_ts_sax)[:, :, 0]

    # 1d-SAX
    arr_symb_ts_onedsax = one_d_sax.fit_transform(X_train_scaled, scale=False)
    arr_symb_ts_onedsax_inv = one_d_sax.inverse_transform(arr_symb_ts_onedsax)[
        :, :, 0]

    # SFA
    M, quantile = train_SFA(
        X_train_scaled[:, :, 0], n_segments_mean//2, n_symbols)
    arr_symb_ts_sfa_inv = test_SFA(
        X_train_scaled[:, :, 0], n_segments_mean//2, n_symbols, M, quantile)

    # ASTRIDE
    astride.fit(X_train_scaled)
    _, _, astride_features_with_symbols_df = transform_symb_ts(
        astride, X_train_scaled)
    arr_symb_ts_astride_inv = inv_transform_symb_ts(
        features_with_symbols_df=astride_features_with_symbols_df,
    )

    # FASTRIDE
    fastride.fit(X_train_scaled)
    _, _, fastride_features_with_symbols_df = transform_symb_ts(
        fastride, X_train_scaled)
    arr_symb_ts_fastride_inv = inv_transform_symb_ts(
        features_with_symbols_df=fastride_features_with_symbols_df,
    )

    # Get the reconstructed signals for all methods
    d_reconstructed_signals = dict()
    d_reconstructed_signals["SAX"] = arr_symb_ts_sax_inv
    d_reconstructed_signals["1d-SAX"] = arr_symb_ts_onedsax_inv
    d_reconstructed_signals["SFA"] = arr_symb_ts_sfa_inv
    d_reconstructed_signals["ABBA"] = arr_symb_ts_abba_inv
    d_reconstructed_signals["ASTRIDE"] = arr_symb_ts_astride_inv
    d_reconstructed_signals["FASTRIDE"] = arr_symb_ts_fastride_inv

    # Check the shape of the reconstructed signals
    for method_name in d_reconstructed_signals:
        err_msg = (
            f"The reconstructed signals from {method_name} do not have the"
            "right shape."
        )
        assert d_reconstructed_signals[method_name].shape == X_train.shape, err_msg

    # ==========================================================================
    # Compute the reconstruction errors
    # Including non-truncation or not, depending on `is_compute_error_not_truncated`

    if is_compute_error_not_truncated:
        df_reconstruction_not_trunc_errors = compute_reconstruction_errors(
            X_signals=X_train_scaled,
            d_reconstructed_signals=d_reconstructed_signals,
        )

    # Truncate the original and reconstructed signals
    X_train_scaled_trunc = truncate_signals(X_train_scaled, n_segments_mean)
    d_reconstructed_signals_trunc = dict()
    for method in d_reconstructed_signals:
        d_reconstructed_signals_trunc[method] = truncate_signals(
            d_reconstructed_signals[method], n_segments_mean)

    df_reconstruction_trunc_errors = compute_reconstruction_errors(
        X_signals=X_train_scaled_trunc,
        d_reconstructed_signals=d_reconstructed_signals_trunc,
    )
    df_reconstruction_trunc_errors.rename(
        columns={"eucl_error": "trunc_eucl_error",
                 "dtw_error": "trunc_dtw_error"},
        inplace=True
    )

    if is_compute_error_not_truncated:
        df_reconstruction_alltrunc_errors = (
            df_reconstruction_not_trunc_errors
            .merge(
                df_reconstruction_trunc_errors,
                on=["method", "signal_index"],
                how="left"
            )
        ).copy()
    else:
        df_reconstruction_alltrunc_errors = df_reconstruction_trunc_errors.copy()

    df_reconstruction_errors_all = (
        df_reconstruction_alltrunc_errors
        .merge(df_tol_abba, on=["signal_index"], how="left")
    )

    df_reconstruction_errors_all["n_samples"] = n_samples
    df_reconstruction_errors_all["memory_usage_ratio_perc"] = (
        df_reconstruction_errors_all.n_segments /
        df_reconstruction_errors_all.n_samples * 100
    )

    # ==========================================================================
    # Export the reconstructions and their errors

    # Make sure the folder exists
    cwd = Path.cwd()
    folder = Path(cwd / "results" / date_exp / "reconstruction" / str(denom))
    folder_reconstructed = Path(
        cwd / "results" / date_exp / "reconstruction" / str(denom) / "reconstructed_signals")
    create_path(folder)
    create_path(folder_reconstructed)

    # Export the reconstructed signals
    d_method_names_abb = {
        "SAX": "sax",
        "1d-SAX": "onedsax",
        "SFA": "sfa",
        "ABBA": "abba",
        "ASTRIDE": "astride",
        "FASTRIDE": "fastride",
    }
    for method_name in d_reconstructed_signals:
        arr = np.array(d_reconstructed_signals[method_name])
        method_name_abb = d_method_names_abb[method_name]
        file_name = folder_reconstructed / \
            f"reconstructed_{dataset_name_ucr}_{method_name_abb}"
        np.save(file_name, arr)

    # Export the reconstruction errors
    df_reconstruction_errors_all.to_csv(
        folder / f"reconstruction_errors_{dataset_name_ucr}.csv",
        index=False,
    )

    b_reconstruction_errors = Bunch(
        d_reconstructed_signals=d_reconstructed_signals,
        df_reconstruction_errors=df_reconstruction_errors_all,
    )
    return b_reconstruction_errors


def load_reconstruction_errors(list_denoms, date_exp):
    """The reconstruction errors are on the truncated signals.
    """

    l_df_reconstruct_err_alldatasets_denom = list()

    for denom in list_denoms:
        path_denom = Path(cwd / "results" / date_exp /
                          "reconstruction" / str(denom))
        l_df_reconstruct_err_all_methods_denom = list(
            path_denom.rglob("reconstruction_errors_*.csv"))
        df_reconstruct_err_alldatasets_denom = concatenate_df(
            l_df_reconstruct_err_all_methods_denom).drop_duplicates()
        l_df_reconstruct_err_alldatasets_denom.append(
            df_reconstruct_err_alldatasets_denom)
    df_reconstruct_err_alldatasets_alldenoms = (
        pd.concat(l_df_reconstruct_err_alldatasets_denom, ignore_index=True)
    )

    df_errors = df_reconstruct_err_alldatasets_alldenoms.copy()
    d_rename = {"trunc_eucl_error": "euclidean_error",
                "trunc_dtw_error": "dtw_error"}
    df_errors = df_reconstruct_err_alldatasets_alldenoms.rename(
        columns=d_rename).copy()

    return df_errors


def get_original_and_reconstructed_signals(
    dataset_name_ucr,
    denom,
    d_method_names_abb,
    df_errors,
    date_exp,
):
    b_load_ucr_dataset = load_ucr_dataset(dataset_name_ucr)
    X_train = b_load_ucr_dataset.X_train

    n_segments = df_errors.query(f"dataset == '{dataset_name_ucr}' and denom == {denom}")[
        "n_segments"].unique()[0]

    scaler = TimeSeriesScalerMeanVariance()
    X_train_scaled = scaler.fit_transform(X_train)

    path_reconstructed = Path(cwd / "results" / date_exp /
                              "reconstruction" / str(denom) / "reconstructed_signals")
    d_reconstructed_signals = dict()
    for method in d_method_names_abb:
        method_name_abb = d_method_names_abb[method]
        file_path = path_reconstructed / \
            f"reconstructed_{dataset_name_ucr}_{method_name_abb}.npy"
        arr_reconstructed_signals_method = np.load(
            file_path, allow_pickle=False)
        d_reconstructed_signals[method] = arr_reconstructed_signals_method

    X_train_scaled_trunc = truncate_signals(X_train_scaled, n_segments)

    d_reconstructed_signals_trunc = dict()
    for method in d_reconstructed_signals:
        d_reconstructed_signals_trunc[method] = truncate_signals(
            d_reconstructed_signals[method], n_segments)

    b_get_original_and_reconstructed_signals = Bunch(
        dataset_name_ucr=dataset_name_ucr,
        denom=denom,
        date_exp=date_exp,
        X_train_scaled=X_train_scaled,
        X_train_scaled_trunc=X_train_scaled_trunc,
        d_reconstructed_signals=d_reconstructed_signals,
        d_reconstructed_signals_trunc=d_reconstructed_signals_trunc,
        n_segments=n_segments,
    )
    return b_get_original_and_reconstructed_signals


def plot_original_and_reconstructed_signals(
    denom,
    dataset,
    X_signals,
    d_reconstructed_signals,
    signal_index,
    is_savefig=False,
    fig_name="plots",
):

    fig, axs = plt.subplots(3, 2, figsize=(
        8, 6), constrained_layout=True, sharex=True, sharey=True)
    l_coord = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    original_signal = list(X_signals[signal_index, :, 0])

    for (enum, method_name) in enumerate(d_reconstructed_signals):
        (i, j) = l_coord[enum]
        reconstructed_signal = list(
            d_reconstructed_signals[method_name][signal_index, :])
        axs[i, j].plot(original_signal, label="original signal",
                       linestyle="--", alpha=.8)
        axs[i, j].plot(reconstructed_signal, label="reconstructed signal")
        axs[i, j].legend(fontsize=8)
        axs[i, j].set_title(f"{method_name}")

    plt.tight_layout()
    plt.margins(x=0)
    if is_savefig:
        cwd = Path.cwd()
        folder_path = cwd / "results/img"
        file_name = f"reconstructed_signals_denom{denom}_{dataset}_{signal_index}_{fig_name}.png"
        plt.savefig(folder_path / file_name, dpi=200)
    plt.show()


def plot_and_show_original_and_reconstructed_trunc_or_not_signals(
    b_get_original_and_reconstructed_signals,
    df_errors,
    signal_index,
    is_savefig_trunc=False,
):
    """Also print the reconstruction errors.
    """

    denom = b_get_original_and_reconstructed_signals.denom
    dataset_name_ucr = b_get_original_and_reconstructed_signals.dataset_name_ucr
    X_train_scaled = b_get_original_and_reconstructed_signals.X_train_scaled
    X_train_scaled_trunc = b_get_original_and_reconstructed_signals.X_train_scaled_trunc
    d_reconstructed_signals = b_get_original_and_reconstructed_signals.d_reconstructed_signals
    d_reconstructed_signals_trunc = b_get_original_and_reconstructed_signals.d_reconstructed_signals_trunc
    n_segments = b_get_original_and_reconstructed_signals.n_segments
    print(f"{n_segments = }")

    print("Without truncation:")
    plot_original_and_reconstructed_signals(
        denom=denom,
        dataset=dataset_name_ucr,
        X_signals=X_train_scaled,
        d_reconstructed_signals=d_reconstructed_signals,
        signal_index=signal_index,
    )

    print("With truncation:")
    plot_original_and_reconstructed_signals(
        denom=denom,
        dataset=dataset_name_ucr,
        X_signals=X_train_scaled_trunc,
        d_reconstructed_signals=d_reconstructed_signals_trunc,
        signal_index=signal_index,
        is_savefig=is_savefig_trunc,
    )

    print("Reconstruction errors (with truncation):")
    display(
        df_errors
        .query(f"denom == {denom} and dataset == '{dataset_name_ucr}' and signal_index == {signal_index}")
        [["method", "euclidean_error", "dtw_error"]]
        .style.background_gradient(
            axis=0,
            cmap='YlOrRd',
        ).format(precision=1)
    )
