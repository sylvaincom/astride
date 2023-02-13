import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error


class SegmentFeature(BaseEstimator):
    """
    Computing the features per segment.

    Inputs a list of segmented signals (meaning signals with their according
        list of breakpoints) and a list of features
    Outputs the computed features per segment.

    Parameters
    ----------
    features_names : list, default=['mean']
        Features to compute per segment, list of strings who must belong to the
        following possible values:

        - 'mean',
        - 'min',
        - 'max',
        - 'mean_of_min_max' (which is the mean between the max and the min)
        - 'variance',
        - 'slope',
        - 'scaled_complexity_invariance',
        - 'linear_residuals'.
        - 'length'
    """

    ALL_FEATURE_NAMES = [
        "mean",
        "min",
        "max",
        "mean_of_min_max",
        "variance",
        "slope",
        "scaled_complexity_invariance",
        "linear_residuals",
        "length",
    ]

    def __init__(
        self,
        features_names=[
            "mean",
        ],
    ) -> None:

        # Check that all asked features are in ALL_FEATURE_NAMES
        for feature_name in features_names:
            err_msg = f"Choose an existing feature, not {feature_name}."
            assert feature_name in SegmentFeature.ALL_FEATURE_NAMES, err_msg

        self.features_names = features_names

    def fit(self, *args, **kwargs):
        return self

    def transform(self, b_transform_segmentation) -> pd.DataFrame:
        """Signals are assumed to be 1D."""
        list_of_signal1D = b_transform_segmentation.list_of_signals
        list_of_bkps = b_transform_segmentation.list_of_bkps
        list_of_df = list()
        for (signal_index, (signal, bkps)) in enumerate(
            zip(list_of_signal1D, list_of_bkps)
        ):
            features_for_single_signal = self.transform_single_signal(
                signal=signal, bkps=bkps
            )

            features_for_single_signal_df = pd.DataFrame(
                features_for_single_signal
            ).add_suffix(
                "_feat"
            )  # adding a suffix to feature columns
            features_for_single_signal_df["signal_index"] = signal_index
            features_for_single_signal_df["segment_start"] = [0] + bkps[:-1]
            features_for_single_signal_df["segment_end"] = bkps
            features_for_single_signal_df["segment_length"] = (
                features_for_single_signal_df.segment_end
                - features_for_single_signal_df.segment_start
            )
            list_of_df.append(features_for_single_signal_df)
        segment_features_df = pd.concat(list_of_df).reset_index(drop=True)
        if "length" in self.features_names:
            segment_features_df.insert(
                len(self.features_names)-1,
                "length_feat",
                segment_features_df["segment_length"].values
            )
        return segment_features_df

    def transform_single_signal(self, signal, bkps):
        """Return a list of features for each segment.

        Output is a list (of dict) of length n_segments."""
        features_for_single_signal = [
            self.feature_func(sub_signal.flatten())
            for sub_signal in np.split(signal, bkps[:-1])
        ]
        return features_for_single_signal

    def feature_func(self, sub_signal):
        """Return a dict of features computed on the whole sub-signal.

        Output is a dict of size `n_features`.
        """
        dict_of_features = dict()
        if "mean" in self.features_names:
            dict_of_features["mean"] = self.get_mean(sub_signal)
        if "min" in self.features_names:
            dict_of_features["min"] = self.get_min(sub_signal)
        if "max" in self.features_names:
            dict_of_features["max"] = self.get_max(sub_signal)
        if "mean_of_min_max" in self.features_names:
            dict_of_features["mean_of_min_max"] = self.get_mean_of_min_max(
                sub_signal)
        if "variance" in self.features_names:
            dict_of_features["variance"] = self.get_var(sub_signal)
        if "slope" in self.features_names:
            dict_of_features["slope"] = self.get_slope(sub_signal)
        if "scaled_complexity_invariance" in self.features_names:
            dict_of_features[
                "scaled_complexity_invariance"
            ] = self.get_scaled_complexity_invariance(sub_signal)
        if "linear_residuals" in self.features_names:
            dict_of_features["linear_residuals"] = self.get_linear_residuals(
                sub_signal
            )
        return dict_of_features

    @staticmethod
    def get_mean(sub_signal):
        return sub_signal.mean()

    @staticmethod
    def get_min(sub_signal):
        return sub_signal.min()

    @staticmethod
    def get_max(sub_signal):
        return sub_signal.max()

    @staticmethod
    def get_mean_of_min_max(sub_signal):
        """For E-SAX (Extended SAX)"""
        return (sub_signal.min() + sub_signal.max()) / 2

    @staticmethod
    def get_slope(sub_signal):
        """Return the value of the slope on the sub-signal."""
        n_samples = sub_signal.shape[0]
        return (sub_signal[-1] - sub_signal[0]) / (n_samples - 1)

    @staticmethod
    def get_var(sub_signal):
        return sub_signal.var()

    @staticmethod
    def get_scaled_complexity_invariance(sub_signal):
        """From the CSAX paper, equation 6 and not equation 5."""
        return (np.sqrt((np.diff(sub_signal) ** 2).sum())) / (len(sub_signal)-1)

    @staticmethod
    def get_linear_residuals(sub_signal):
        y_linear = np.linspace(
            start=sub_signal[0], stop=sub_signal[-1], num=len(sub_signal)
        )
        linear_residuals = mean_squared_error(
            y_true=sub_signal, y_pred=y_linear, squared=False
        )
        return linear_residuals
