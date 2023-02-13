from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from src.metadata import l_datasets_classif_bench
from src.segment_feature import SegmentFeature
from src.segmentation import Segmentation
from src.symbolic_signal_distance import SymbolicSignalDistance
from src.symbolization import Symbolization
from src.tslearn_interface import (DistInterfaceOneD, DistInterfaceSAX, MyOneD,
                                   MySAX)
from src.utils_run_classification import launch_grid_search_acc_datasets


def main(method_name, date_exp, l_datasets_to_compute):
    """In the classification benchmark, there are 86 data sets from the UCR
    archive which are univariate, equal-size and have at least 100 samples.
    """

    # Check if the method name is correct
    method_names = [
        "sax",  # our SAX implementation
        "saxtslearn",  # SAX implementation from tslearn
        "1dsax",
        "astride",
        "fastride",
    ]
    err_msg = f"Choose an existing method name, not {method_name}."
    assert method_name in method_names, err_msg

    # Verbose
    print(f"\nMethod name: {method_name}")
    print(f"Date of experiment: {date_exp}")
    n_datasets = len(l_datasets_to_compute)
    print(f"UCR data sets under consideration: {n_datasets}")

    # Define the parameter grid of the hyper-parameters of the symbolization methods
    param_grid_n_segments = [5, 10, 15, 20, 25]
    param_grid_n_symbols = [4, 9, 16, 25]
    param_grid_alphabet_size_avg = [2, 3, 4, 5]  # for 1d-SAX
    param_grid_alphabet_size_slope = [2, 3, 4, 5]  # for 1d-SAX    

    # Define the pipelines for the symbolization methods
    pipe_symbts = make_pipeline(
        TimeSeriesScalerMeanVariance(),
        Segmentation(),
        SegmentFeature(),
        Symbolization(),
        SymbolicSignalDistance(),
        KNeighborsClassifier(n_neighbors=1, metric="precomputed"),
    )
    pipe_tslearn_sax = make_pipeline(
        MySAX(
            scale=True
        ),
        DistInterfaceSAX(),
        KNeighborsClassifier(n_neighbors=1, metric="precomputed"),
    )
    pipe_tslearn_1dsax = make_pipeline(
        MyOneD(
            sigma_l=1.0,
            scale=True,
        ),
        DistInterfaceOneD(),
        KNeighborsClassifier(n_neighbors=1, metric="precomputed"),
    )

    # Define the parameter grid for each symbolization method
    param_grid_sax = {
        "segmentation__univariate_or_multivariate": ["multivariate"],
        "segmentation__uniform_or_adaptive": ["uniform"],
        "segmentation__mean_or_slope": [None],
        "segmentation__n_segments": param_grid_n_segments,
        "segmentation__pen_factor": [None],
        "segmentfeature__features_names": [["mean"]],
        "symbolization__n_symbols": param_grid_n_symbols,
        "symbolization__symb_method": ["quantif"],
        "symbolization__symb_quantif_method": ["gaussian"],
        "symbolization__symb_cluster_method": [None],
        "symbolization__features_scaling": [None],
        "symbolization__reconstruct_bool": [False],
        "symbolization__n_regime_lengths": [None],
        "symbolization__seglen_bins_method": [None],
        "symbolization__lookup_table_type": ["mindist"],
        "symbolicsignaldistance__distance": ["euclidean"],
        "symbolicsignaldistance__n_samples": [None],  # to be set
        "symbolicsignaldistance__weighted_bool": [True],
    }
    param_grid_tslearn_sax = {
        "mysax__n_segments": param_grid_n_segments,
        "mysax__alphabet_size_avg": param_grid_n_symbols,
    }
    param_grid_tslearn_1dsax = {
        "myoned__n_segments": param_grid_n_segments,
        "myoned__alphabet_size_avg": param_grid_alphabet_size_avg,
        "myoned__alphabet_size_slope": param_grid_alphabet_size_slope,
    }
    param_grid_astride = {
        "segmentation__univariate_or_multivariate": ["multivariate"],
        "segmentation__uniform_or_adaptive": ["adaptive"],
        "segmentation__mean_or_slope": ["mean"],
        "segmentation__n_segments": param_grid_n_segments,
        "segmentation__pen_factor": [None],
        "segmentfeature__features_names": [["mean"]],
        "symbolization__n_symbols": param_grid_n_symbols,
        "symbolization__symb_method": ["quantif"],
        "symbolization__symb_quantif_method": ["quantiles"],
        "symbolization__symb_cluster_method": [None],
        "symbolization__features_scaling": [None],
        "symbolization__reconstruct_bool": [True],
        "symbolization__n_regime_lengths": ["divide_exact"],
        "symbolization__seglen_bins_method": [None],
        "symbolization__lookup_table_type": ["mof"],
        "symbolicsignaldistance__distance": ["lev"],
        "symbolicsignaldistance__n_samples": [None],
        "symbolicsignaldistance__weighted_bool": [True],
    }
    param_grid_fastride = {
        "segmentation__univariate_or_multivariate": ["multivariate"],
        "segmentation__uniform_or_adaptive": ["uniform"],
        "segmentation__mean_or_slope": [None],
        "segmentation__n_segments": param_grid_n_segments,
        "segmentation__pen_factor": [None],
        "segmentfeature__features_names": [["mean"]],
        "symbolization__n_symbols": param_grid_n_symbols,
        "symbolization__symb_method": ["quantif"],
        "symbolization__symb_quantif_method": ["quantiles"],
        "symbolization__symb_cluster_method": [None],
        "symbolization__features_scaling": [None],
        "symbolization__reconstruct_bool": [False],
        "symbolization__n_regime_lengths": [None],
        "symbolization__seglen_bins_method": [None],
        "symbolization__lookup_table_type": ["mof"],
        "symbolicsignaldistance__distance": ["lev"],
        "symbolicsignaldistance__n_samples": [None],
        "symbolicsignaldistance__weighted_bool": [True],
    }

    # Select the pipeline and the parameter grid according to the method name
    if method_name == "sax":
        pipe_bench = pipe_symbts
        param_grid_bench = param_grid_sax
    elif method_name == "saxtslearn":
        pipe_bench = pipe_tslearn_sax
        param_grid_bench = param_grid_tslearn_sax
    elif method_name == "1dsax":
        pipe_bench = pipe_tslearn_1dsax
        param_grid_bench = param_grid_tslearn_1dsax
    elif method_name == "fastride":
        pipe_bench = pipe_symbts
        param_grid_bench = param_grid_fastride
    elif method_name == "astride":
        pipe_bench = pipe_symbts
        param_grid_bench = param_grid_astride

    # Launch the classification
    _ = launch_grid_search_acc_datasets(
        l_datasets_bench=l_datasets_to_compute,
        pipe=pipe_bench,
        param_grid=param_grid_bench,
        method_name=method_name,
        date_exp=date_exp,
    )


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method_name",
        type=str,
        help="Name of the method in {'sax', 'saxtslearn', '1dsax', 'astride', 'fastride'}.",
        required=True,
    )
    parser.add_argument(
        "--date_exp",
        type=str,
        help="Date of the launch of the experiments (for versioning).",
        required=True,
    )

    args = parser.parse_args()
    main(args.method_name, args.date_exp, l_datasets_classif_bench)