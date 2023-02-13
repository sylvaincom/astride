from pathlib import Path

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
from scipy.stats import normaltest
from sklearn.utils import Bunch
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from src.segment_feature import SegmentFeature
from src.segmentation import Segmentation
from src.utils import create_path, load_ucr_dataset


def run_gaussian_assumption(
    dataset_name_ucr: str,
):
    """
    Checking the Gaussian assumption in SAX for a single data set.
    The word length w is automatically determined such that n >= 4 * w
        with w as high as possible in {2, 4, 8, 16, 32, 64}.

    Parameters
    ----------
    dataset_name_ucr : str
        Name of the UCR data set considered.

    print_bool: bool, default=False
        Boolean deciding if we print the results or not. 
    """

    # Load the data set
    b_load_ucr_dataset = load_ucr_dataset(dataset_name_ucr)
    l_train_test = b_load_ucr_dataset.l_train_test
    n_samples = len(l_train_test[0])

    # Get the word length `n_segments` (that will be used in the segmentation)
    i = 6
    n_segments = 2**i
    while n_samples < 4 * n_segments:
        i -= 1
        n_segments = 2**i

    # Normalize the data
    l_train_test = TimeSeriesScalerMeanVariance().fit_transform(l_train_test)

    # Perform the segmentation
    seg = Segmentation(
        univariate_or_multivariate="univariate",
        uniform_or_adaptive='uniform',
        mean_or_slope=None,
        n_segments=n_segments,
        pen_factor=None
    )
    b_transform_segmentation = seg.fit(l_train_test).transform(l_train_test)

    # Compute the means per segment
    features_names = ["mean"]
    segment_features_df = SegmentFeature(features_names).fit().transform(
        b_transform_segmentation=b_transform_segmentation
    )

    # Perform the statistical test for the normality assumption
    x = segment_features_df["mean_feat"].tolist()
    _, p_value = normaltest(x)
    alpha = 0.05
    if p_value < alpha:  # null hypothesis: `x` comes from a normal distribution
        res_H0 = "rejected"
    else:
        res_H0 = "cannot be rejected"

    b_get_means_per_segments = Bunch(
        segment_features_df=segment_features_df,
        n_samples=n_samples,
        n_segments=n_segments,
        p_value=p_value,
        res_H0=res_H0,
    )
    return b_get_means_per_segments

def explore_dataset_means(
    dataset_name_ucr,
    all_segment_features_df,
    df_normality_test,
    is_display_title=False,
    is_save_fig=False,
    date_exp="unknown",
    kde_bool=False
):
    """
    Focusing on a specific data set.
    """
    
    df_normality_test_dataset = df_normality_test.query(f"dataset == '{dataset_name_ucr}'")
    n_samples = df_normality_test_dataset["n_samples"].tolist()[0]
    n_segments = df_normality_test_dataset["n_segments"].tolist()[0]
    p_value = df_normality_test_dataset["p_value"].tolist()[0]
    
    segment_features_df = all_segment_features_df.query(f"dataset == '{dataset_name_ucr}'")
    
    # Percentage of NaN mean values
    perc_nan = segment_features_df["mean_feat"].isna().sum() / len(segment_features_df) * 100
    print(f"For the {dataset_name_ucr} data set, there are {round(perc_nan, 2)}% of NaN mean values.")
    
    # Histogram of the means per segment
    plt.figure(figsize=(8, 4))
    sns.histplot(data=segment_features_df, x="mean_feat")
    str_title = (
        f"Data set: {dataset_name_ucr}."
        f"\nNumber of samples per signal: {n_samples}. "
        f"Number of uniform segments: {n_segments}."
        f"\np-value for the normality test: {p_value}."
    )
    plt.xlabel("Mean per segment")
    plt.margins(x=0, y=0)
    plt.tight_layout()
    if is_display_title:
        plt.title(str_title, loc="left")
    if is_save_fig:
        cwd = Path.cwd()
        folder = cwd / f"results/{date_exp}/img"
        create_path(folder)
        plt.savefig(folder / f"gaussian_assumption_{dataset_name_ucr}.png", dpi=200)
    plt.show()
    
    # KDE plot of the means per segment
    if kde_bool:
        hist_data = [segment_features_df["mean_feat"]]
        group_labels = [f"{dataset_name_ucr}"] # name of the dataset
        fig = ff.create_distplot(hist_data, group_labels)
        fig.update_layout(title=f"Histogram, kde plot and rug plot of the mean per segments feature.")
        fig.show()
    
    return df_normality_test_dataset