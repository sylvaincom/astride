import random

import numpy as np
import pandas as pd
from joblib import cpu_count
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from sklearn.base import BaseEstimator


class Symbolization(BaseEstimator):
    """
    Attributing a symbol per segment.

    Inputs the computed features per segment, there can be more than one
        feature.
    Outputs the symbolic signals (multivariate or univariate) as well as the
        (unique or several) look-up table which is the pairwise distance matrix
        between all the individual symbols.
    Amounts to discretizing our signals along the y-axis.

    Parameters
    ----------

    n_symbols : int, default=5
        Number of possible unique symbols for our symbolic representation.
        Corresponds to the size of alphabet in the case of Vanilla SAX.

        Must be inferior or equal to 26 which is the size of the English
            alphabet, e.g. the number of possible letters. Indeed, if we
            want to use the weighted Levenshtein distance on our symbolic
            signals, we will convert our integer symbols to letter symbols.

    symb_method : {'quantif', 'cluster'}, default='quantif'
        Family of method for the symbolization. Possible values:

        - 'quantif' : symbolization using quantification along
            the y-axis.
        - 'cluster' : symbolization through clustering of the computed
            features per segment.

    symb_quantif_method : {'gaussian', 'quantiles', None}, default='gaussian'
        Quantification method for the symbolization when `symb_method` is 'quantif'.
        Possible values:

        - 'gaussian' : quantification using Gaussian breakpoints.
            In this case, the symbolization is the same as performed in
            vanilla SAX.
        - 'quantiles' : quantification using quantiles as breakpoints.
        - None : `symb_mehtod` is not `quantif`.

    symb_cluster_method : {'kmeans', 'kmeans_partial', 'minibatch_kmeans', None}, default='None'
        Clustering method for the symbolization when `symb_method` is 'cluster'.
        Possible values:

        - 'kmeans' : regular K-means.
        - 'kmeans_partial' : K-means using 20% of the input size.
        - 'minibatch_kmeans' : mini-batch K-means using 20% of the input size.
        - None : `symb_method` is not 'cluster'.

    features_scaling : dict, default='None'
        For symbolization using clustering, multiplicative coefficient in front
            a feature (after z-normalization).

    numerosity_reduction : bool, default=False
        False when no numerosity reduction is applied (before eventual
            reconstruction).

    reconstruct_bool : bool, default=False
        False when no reconstruction of our symbolic signals is done.

        When doing uniform segmentation, no need for reconstruction.
        When doing adaptive segmentation, we recommend using reconstruction.

    n_regime_lengths : int or list or str or None, default=None
        When the type is int, `n_regime_lengths` is the number of possible
            unique regime length values when performing reconstruction of
            our symbolic signals.
            Amounts to discretizing or quantifying the regime lengths (segment
            lengths).
            The quantified regime lengths are divided by their minimum then
            rounded.
        When the type is list, it must be of length `n_symbols`, there is a
            specific value of `n_regime_lengths` per symbol.
            Hence, the quantified regime lengths values are different from a
            symbol to another.
            Note that we must get our segment symbols first.
            The quantified regime lengths (of all symbols) are divided by
            their minimum then rounded.
        If `n_regime_lengths` is `'ccl'`: when doing symbolization with
            clustering on features including the segment length, the quantified
            segment lengths are the scaled rounded corresponding coordinates
            of the cluster centers.
        If `n_regime_lengths` is `'divide_exact'`: when doing adaptive
            multivariate segmentation, divide the `n_segments` segment lengths
            by their minimum.
        If None and we are performing reconstruction of our symbolic
            signals, then no quantification of the regime lengths is done.
        When doing adaptive segmentation and reconstruction, we recommend
            quantifying the regime lengths for memory purposes (and not for
            performance purposes).

    seglen_bins_method : {'linspace', 'quantiles', None}, default=None
        Method to obtain the segment bins `self.seglen_bins_` for the
        quantification of the regime lengths.
        Possible values:

        - 'linspace' : obtaining the `self.seglen_bins_`  with evenly spaced
            bins over the minimum and maximum of segment lengths.
        - 'quantiles' : obtaining the `self.seglen_bins_` using quantiles.
        - None : if `n_regime_lengths` is `None`

    lookup_table_type : {'mindist', 'mof', 'eucl_cc', 'eucl_ccm', None}, default='mindist'
        Type of distance between pairwise individual symbols which will be
            used to build the look-up table. Possible values:

        - 'mindist' : MINDIST as defined in Vanilla SAX (we assume that we only
            have one feature per segment).
        - 'mof' : mean of feature (we assume that we only have one feature per
            segment).
            The distance between two symbols is the distance between the mean
                of all the values of the feature for a same symbol.
        - 'eucl_cc' : only when the symbolization method is clustering.
            The distance between two symbols is the euclidean distance between
                their corresponding cluster centers.
        - 'eucl_ccm' : only when the symbolization method is clustering.
            The distance between two symbols is the euclidean distance between
                their corresponding cluster centers whose coordinates have
                been truncated to the mean feature only.

    Attributes
    ----------
    scaler_ : fitted scikit-learn model (sklearn.preprocessing.StandardScaler)
        When the symbolization is done through clustering of the features
            per segment, we first need to scale our data.

    clustering_model_ : fitted scikit-learn model (e.g. sklearn.cluster.KMeans)
        When the symbolization is done through clustering.

    scaled_cluster_centers_df_

    unscaled_cluster_centers_df_

    lookup_table_ : ndarray of shape (n_symbols, n_symbols)
        Pairwise distance matrix between all the individual symbols.

    y_quantif_bins_ : list of length (n_symbols-1)
        When the symbolization method is quantification: bins for the
            quantification of the unique feature.

    seglen_bins_ : (list of) list of length (n_regime_lengths-1)
        When doing reconstruction and quantifying the segment lengths, bins
            for the quantification of the obtained segment lengths.

    from_seglen_label_to_value_dict_ : (list of) dict of length (n_regime_lengths)
        When doing reconstruction and quantifying the segment lengths, mapping
            the quantified segment lengths values to the mean of the real
            segment lengths that got the same quantified value.

    from_cluster_label_to_seglenquantif_dict_ : dict of length `n_symbols`
        When doing symbolization with clustering on features including the
            segment length.
        The quantified regime lengths are the cluster centers coordinates
            corresponding to the regime lengths.
        The quantified regime lengths are divided by their minimum then rounded.
    """

    def __init__(
        self,
        n_symbols: int = 5,
        symb_method="quantif",
        symb_quantif_method="gaussian",
        symb_cluster_method=None,
        features_scaling=None,
        numerosity_reduction: bool = False,
        reconstruct_bool: bool = False,
        n_regime_lengths=None,
        seglen_bins_method=None,
        lookup_table_type="mindist",
    ) -> None:

        # Unit tests on the parameters:

        # err_msg = (
        #     f"`n_symbols` must be an integer lower than 26 because the "
        #     "alphabet has 26 letters, and not {n_symbols}."
        # )
        # assert type(n_symbols) == int and n_symbols <= 26, err_msg
        err_msg = (
            f"`n_symbols` must be an integer, not {n_symbols}."
        )
        assert type(n_symbols) == int, err_msg

        err_msg = (
            "`numerosity_reduction` must be a boolean, and not "
            f"{numerosity_reduction}."
        )
        assert type(numerosity_reduction) == bool, err_msg

        err_msg = (
            "`reconstruct_bool` must be a boolean, and not "
            f"{reconstruct_bool}."
        )
        assert type(reconstruct_bool) == bool, err_msg

        err_msg = (
            "Choose between quantification (`quantif`) or clustering "
            f"(`cluster`), not {symb_method}."
        )
        assert symb_method in ["quantif", "cluster"], err_msg

        if symb_method == "quantif":

            err_msg = (
                "If the symbolization is done with quantification, choose "
                f"`gaussian` or `quantiles`, not {symb_quantif_method}."
            )
            assert symb_quantif_method in ["gaussian", "quantiles"], err_msg

            err_msg = (
                "If the symbolization is done with quantification, "
                f"`symb_cluster_method` should be None, not {symb_cluster_method}."
            )
            assert symb_cluster_method is None, err_msg

            err_msg = (
                "If the symbolization is done with quantification, "
                f"`features_scaling` should be None, not {features_scaling}."
            )
            assert features_scaling is None, err_msg

            err_msg = (
                "If the symbolization is done with quantification, choose "
                f"`mindist` or `mof`, not {lookup_table_type}."
            )
            assert lookup_table_type in ["mindist", "mof"], err_msg

        if symb_method == "cluster":

            err_msg = (
                "If the symbolization is done with clustering, choose "
                "`kmeans` or `kmeans_partial` or `minibatch_kmeans`, "
                f"not {symb_cluster_method}."
            )
            assert symb_cluster_method in [
                "kmeans", "kmeans_partial", "minibatch_kmeans"], err_msg

            err_msg = (
                "If the symbolization is done with clustering, "
                f"`symb_quantif_method` should be None, not {symb_quantif_method}."
            )
            assert symb_quantif_method is None, err_msg

            err_msg = (
                "If the symbolization is done with clustering, choose "
                f"`eucl_cc` or `eucl_ccm`, not {lookup_table_type}."
            )
            assert lookup_table_type in ["eucl_cc", "eucl_ccm"], err_msg

            if features_scaling is not None:
                err_msg = (
                    "If the symbolization is done with clustering, "
                    "and `features_scaling` is not `None`, it must be a "
                    "dictionary."
                )
                assert type(features_scaling) == dict, err_msg

                if type(features_scaling) == dict:
                    for key in features_scaling:
                        err_msg = (
                            "If the symbolization is done with clustering, "
                            "the keys of `features_scaling` must be strings."
                        )
                        assert type(key) == str, err_msg
                        err_msg = (
                            "If the symbolization is done with clustering, "
                            "the values of `features_scaling` must be positive "
                            "floats."
                        )
                        assert features_scaling[key] >= 0, err_msg

        if not reconstruct_bool:

            err_msg = (
                f"If there is no reconstruction, `n_regime_lengths` should be "
                f"`None`, not {n_regime_lengths}."
            )
            assert n_regime_lengths is None, err_msg

            err_msg = (
                f"If there is no reconstruction, `seglen_bins_method` should be "
                f"`None`, not {seglen_bins_method}."
            )
            assert seglen_bins_method is None, err_msg

        else:  # there is reconstruction

            # err_msg = (
            #     "If there is reconstruction, `n_regime_lengths` should not be "
            #     "`None`"
            # )
            # assert n_regime_lengths is not None, err_msg

            if type(n_regime_lengths) == list:
                err_msg = (
                    "`n_regime_lengths` must be a list of length `n_symbols`, "
                    f"and not {len(n_regime_lengths)}."
                )
                assert len(n_regime_lengths) == n_symbols, err_msg

            err_msg = (
                "If there is reconstruction, for `seglen_bins_method` choose "
                f"`linspace` or `quantiles` or `None`, {seglen_bins_method}."
            )
            assert seglen_bins_method in [
                "linspace", "quantiles", None], err_msg

        # Initializing the parameters
        self.n_symbols = n_symbols
        self.symb_method = symb_method
        self.symb_quantif_method = symb_quantif_method
        self.symb_cluster_method = symb_cluster_method
        self.features_scaling = features_scaling
        self.numerosity_reduction = numerosity_reduction
        self.reconstruct_bool = reconstruct_bool
        self.n_regime_lengths = n_regime_lengths
        self.seglen_bins_method = seglen_bins_method
        self.lookup_table_type = lookup_table_type
        self.scaler_ = None
        self.clustering_model_ = None
        self.scaled_cluster_centers_df_ = None
        self.unscaled_cluster_centers_df_ = None
        self.lookup_table_ = None
        self.y_quantif_bins_ = None
        self.seglen_bins_ = None
        self.from_seglen_label_to_value_dict_ = None
        self.from_cluster_label_to_seglenquantif_dict_ = None

    def fit(self, segment_features_df: pd.DataFrame, *args, **kwargs):

        # if `symb_method` is quantification, there are two possible ways to compute the
        # lookup table: MINDIST type or mean of univariate feature (MoF).
        if self.symb_method == "quantif":
            err_msg = "Choose a lookup table type."
            assert self.lookup_table_type is not None, err_msg
            err_msg = f"Choose a valid lookup table type, not {self.lookup_table_type}."
            assert self.lookup_table_type in ["mindist", "mof"], err_msg

        # After unit testing, the proper fit occurs here:
        if self.reconstruct_bool and self.n_regime_lengths is not None:
            self.fit_quantif_seglen(
                segment_features_df=segment_features_df
            )
        if self.symb_method == "quantif":
            return self.fit_quantif(segment_features_df=segment_features_df)
        if self.symb_method == "cluster":
            return self.fit_clustering(segment_features_df=segment_features_df)

    def fit_quantif_seglen(self, segment_features_df: pd.DataFrame):
        """Fit the segment lengths' quantification step (limits of bins and
        seglen labels).

        Make sure that `segment_features_df` is a pd.DataFrame.
        """

        err_msg = (
            "`segment_features_df` is a pd.DataFrame, "
            f"not {type(segment_features_df)}."
        )
        assert type(segment_features_df) == pd.DataFrame, err_msg

        if type(self.n_regime_lengths) == int:
            b_get_quantif_seglen = self.get_quantif_seglen(
                segment_lengths=segment_features_df.segment_length,
                n_regime_lengths=self.n_regime_lengths,
                seglen_bins_method=self.seglen_bins_method,
            )
            self.seglen_bins_ = b_get_quantif_seglen.seglen_bins
            self.from_seglen_label_to_value_dict_ = b_get_quantif_seglen.from_seglen_label_to_value_dict
        elif type(self.n_regime_lengths) == list:
            # To be filled after transformation, once we have the segment symbols.
            self.seglen_bins_ = dict()
            self.from_seglen_label_to_value_dict_ = dict()

        return self

    @staticmethod
    def get_quantif_seglen(
        segment_lengths: pd.Series,
        n_regime_lengths: int,
        seglen_bins_method: str
    ):
        """Get the segment lengths' quantification step (limits of bins and
        seglen labels).
        """

        err_msg = (
            f"`n_regime_lengths` must be an integer, not {type(n_regime_lengths)}."
        )
        assert type(n_regime_lengths) == int, err_msg

        err_msg = (
            "`segment_lengths` is a pd.Series or pd.DataFrame, "
            f"not {type(segment_lengths)}."
        )
        assert type(segment_lengths) is pd.Series or pd.DataFrame, err_msg

        # Get the bins
        if seglen_bins_method == "linspace":
            seglen_bins = np.linspace(
                start=min(segment_lengths),
                stop=max(segment_lengths),
                num=n_regime_lengths,
                endpoint=False,
            )[1:]
        elif seglen_bins_method == "quantiles":
            quantiles = np.linspace(
                start=0, stop=1, num=n_regime_lengths+1, endpoint=True)
            seglen_bins = pd.Series(segment_lengths).quantile(
                quantiles).round(0).astype(int).values[1:-1].flatten()
        else:
            err_msg = (
                "`seglen_bins_method` is not well defined."
            )
            assert False, err_msg

        # Get the bin label for the quantified segment lengths
        labeled_segment_lengths = np.digitize(
            segment_lengths,
            bins=seglen_bins,
        )

        # Associate each segment length bin with the median value found in the
        # training set.
        from_seglen_label_to_value_dict = (
            segment_lengths.groupby(labeled_segment_lengths).median().round(
                0).astype(int).to_dict()
        )

        b_get_quantif_seglen = Bunch(
            seglen_bins=seglen_bins,
            from_seglen_label_to_value_dict=from_seglen_label_to_value_dict,
        )
        return b_get_quantif_seglen

    @staticmethod
    def get_feat_df(segment_features_df: pd.DataFrame) -> pd.DataFrame:
        """Return the same df with only the feature columns."""
        feat_columns = [
            col for col in segment_features_df.columns if col.endswith("_feat")
        ]
        return segment_features_df[feat_columns]

    def fit_quantif(self, segment_features_df: pd.DataFrame):
        """Find the bins' limits for the quantification and compute the look-up
        table.

        This function assumes that there is only one feature.
        """

        # Retrieve features
        only_features_df = self.get_feat_df(
            segment_features_df=segment_features_df
        )
        err_msg = (
            "There are more than one feature; not possible with symbolization "
            "using quantification."
        )
        assert only_features_df.shape[1] == 1, err_msg

        # Get bins' limits
        if self.symb_quantif_method == "gaussian":
            self.y_quantif_bins_ = norm.ppf(
                [float(i) / self.n_symbols for i in range(1, self.n_symbols)],
                scale=1,
            )
        elif self.symb_quantif_method == "quantiles":
            quantiles = np.linspace(
                start=0, stop=1, num=self.n_symbols+1, endpoint=True)
            self.y_quantif_bins_ = only_features_df.quantile(
                quantiles).values[1:-1].flatten()

        # Compute look-up table
        if self.lookup_table_type == "mindist":
            self.lookup_table_ = self.compute_lookup_table_mindist(
                y_quantif_bins=self.y_quantif_bins_
            )
        elif self.lookup_table_type == "mof":
            segment_symbols = self.transform_quantif(
                segment_features_df=only_features_df
            )
            feature_1D = only_features_df.values
            self.lookup_table_ = self.compute_lookup_table_mof(
                segment_symbols=segment_symbols, feature_1D=feature_1D
            )

        return self

    def fit_clustering(self, segment_features_df: pd.DataFrame):

        # Retrieve features
        only_features_df = self.get_feat_df(
            segment_features_df=segment_features_df
        )
        # Scaling:
        self.scaler_ = StandardScaler().fit(only_features_df)
        scaled_features = self.scaler_.transform(only_features_df)
        scaled_features_df = pd.DataFrame(
            scaled_features, columns=self.scaler_.feature_names_in_)

        # NEW for SAX-DD-ML-v3
        if self.features_scaling is not None:
            scaled_features_df["length_feat"] = scaled_features_df["length_feat"] * \
                self.features_scaling["length_feat"]

        # Fit the clustering model:
        batch_size_clustering = int(round(0.2 * len(scaled_features_df), 0))
        if self.symb_cluster_method == "kmeans":
            self.clustering_model_ = KMeans(
                n_clusters=self.n_symbols, init="k-means++", random_state=0
            ).fit(scaled_features_df)
        elif self.symb_cluster_method == "kmeans_partial":
            scaled_features_shuffled_df = scaled_features_df.copy()
            random.seed(0)
            np.random.shuffle(scaled_features_shuffled_df)
            self.clustering_model_ = KMeans(
                n_clusters=self.n_symbols, init="k-means++", random_state=0
            ).fit(scaled_features_shuffled_df[0:batch_size_clustering, :])
        elif self.symb_cluster_method == "minibatch_kmeans":
            self.clustering_model_ = MiniBatchKMeans(
                init="k-means++",
                n_clusters=self.n_symbols,
                batch_size=batch_size_clustering,
                n_init=10,
                max_no_improvement=10,
                verbose=0,
                random_state=0,
            ).fit(scaled_features_df)

        # Get the cluster centers, scaled or unscaled
        # NEW for SAX-DD-ML-v3
        if self.features_scaling is not None:
            # The scaling coefficient was only needed to obtain the clusters,
            # but let's go back to cluster centers without the coeff
            scaled_features_df_new = scaled_features_df.copy()
            scaled_features_df_new.length_feat = (
                scaled_features_df.length_feat / self.features_scaling["length_feat"]
            )
            scaled_features_df_new["segment_symbol"] = self.clustering_model_.labels_
            scaled_cluster_centers = (
                scaled_features_df_new.groupby("segment_symbol").mean()
                .reset_index().sort_values(by="segment_symbol")
                .drop(columns=["segment_symbol"])
            )
            #scaled_cluster_centers = self.clustering_model_.cluster_centers_
        else:
            scaled_cluster_centers = self.clustering_model_.cluster_centers_
        self.scaled_cluster_centers_df_ = pd.DataFrame(
            scaled_cluster_centers,
            columns=self.clustering_model_.feature_names_in_
        )
        unscaled_cluster_centers = self.scaler_.inverse_transform(
            scaled_cluster_centers)
        self.unscaled_cluster_centers_df_ = pd.DataFrame(
            unscaled_cluster_centers,
            columns=self.clustering_model_.feature_names_in_
        )

        # Compute the look-up table (and eventually the quantified regime lengths):
        # TODO: maybe unscale the cluster centers value
        if self.lookup_table_type == "eucl_cc":
            self.lookup_table_ = squareform(
                pdist(scaled_cluster_centers)
            )
        elif self.lookup_table_type == "eucl_ccm":

            # Careful: the cluster centers are unscaled here
            self.lookup_table_ = squareform(
                pdist(self.unscaled_cluster_centers_df_[
                      "mean_feat"].to_numpy().reshape(-1, 1))
            )

            # Quantification of the segment lengths
            self.from_cluster_label_to_seglenquantif_dict_ = dict()
            l_quantif_len = self.unscaled_cluster_centers_df_[
                "length_feat"].tolist()
            l_scl_quantif_len = [elem / min(l_quantif_len)
                                 for elem in l_quantif_len]
            for (i, length) in enumerate(l_scl_quantif_len):
                self.from_cluster_label_to_seglenquantif_dict_[i] = round(length)

        return self

    def transform(self, segment_features_df: pd.DataFrame):

        # Transform to symbols and get them
        if self.symb_method == "quantif":
            segment_symbols = self.transform_quantif(
                segment_features_df=segment_features_df
            )
        if self.symb_method == "cluster":
            segment_symbols = self.transform_clustering(
                segment_features_df=segment_features_df
            )
        features_with_symbols_df = segment_features_df.assign(
            segment_symbol=segment_symbols
        ).sort_values(["signal_index", "segment_start"])

        # Without numerosity reduction and without quantification of the
        # segment lengths (and without reconstruction of course)
        _features_with_symbols_nonumreduc_noquantifseglen_df = features_with_symbols_df.copy()

        # Numerosity reduction (or not)
        if self.numerosity_reduction:
            features_with_symbols_df = self.transform_numerosity_reduction(
                features_with_symbols_df=features_with_symbols_df)

        # With (eventual) numerosity reduction and without quantification of the
        # segment lengths (and without reconstruction of course)
        _features_with_symbols_noquantifseglen_df = features_with_symbols_df.copy()

        # Reconstruction (or not)
        if self.reconstruct_bool:  # reconstruction
            # Quantification of the regime lengths (or not)
            if self.n_regime_lengths is not None:
                # Replacing (inplace) the `segment_lengths` column by the
                # quantified version.

                if self.n_regime_lengths == "ccl":
                    features_with_symbols_df.segment_length = features_with_symbols_df.segment_symbol.astype(
                        int).map(self.from_cluster_label_to_seglenquantif_dict_)

                elif type(self.n_regime_lengths) == int:
                    features_with_symbols_df.segment_length = (
                        self.transform_quantif_seglen(
                            segment_lengths=features_with_symbols_df.segment_length
                        )
                    )

                elif type(self.n_regime_lengths) == list:

                    err_msg = "`segment_symbol` must be a feature"
                    assert "segment_symbol" in features_with_symbols_df.columns, err_msg

                    # TODO: why do we need to initialize again, it is already
                    # done in the fit, which is weird
                    l_groups = list()
                    self.seglen_bins_ = dict()
                    self.from_seglen_label_to_value_dict_ = dict()

                    for (segment_symbol, group) in features_with_symbols_df.groupby(by=["segment_symbol"]):

                        b_get_quantif_seglen = self.get_quantif_seglen(
                            segment_lengths=group.segment_length,
                            n_regime_lengths=self.n_regime_lengths[segment_symbol],
                            seglen_bins_method=self.seglen_bins_method,
                        )
                        self.seglen_bins_[
                            segment_symbol] = b_get_quantif_seglen.seglen_bins
                        self.from_seglen_label_to_value_dict_[
                            segment_symbol] = b_get_quantif_seglen.from_seglen_label_to_value_dict

                        group.segment_length = self.apply_quantif_seglen(
                            segment_lengths=group.segment_length,
                            seglen_bins=self.seglen_bins_[segment_symbol],
                            from_seglen_label_to_value_dict=self.from_seglen_label_to_value_dict_[
                                segment_symbol],
                        )
                        l_groups.append(group)
                    features_with_symbols_df = pd.concat(
                        l_groups, ignore_index=True)

                    # Reduce the quantified segment lengths (to make the symbolic
                    # signals shorter)
                    features_with_symbols_df.segment_length = self.shorten_quantif_seglen(
                        features_with_symbols_df.segment_length)
                
                elif self.n_regime_lengths == "divide_exact":
                    # Reduce the quantified segment lengths (to make the symbolic
                    # signals shorter)
                    features_with_symbols_df.segment_length = self.shorten_quantif_seglen(
                        features_with_symbols_df.segment_length)

            # Performing the reconstruction (whether the segment lengths are
            # quantified or not)
            list_of_symbolic_signals = list()
            for (_, group) in features_with_symbols_df.groupby("signal_index"):
                list_of_symbolic_signals.append(
                    np.array(
                        group.segment_symbol.apply(lambda x: [x])
                        * group.segment_length.astype(int)
                    ).sum()
                )
        else:  # no reconstruction (staying in the reduced space)
            list_of_symbolic_signals = (
                features_with_symbols_df.groupby("signal_index")
                .apply(lambda df: df.segment_symbol.to_numpy())
                .tolist()
            )
        b_transform_symbolization = Bunch(
            list_of_symbolic_signals=list_of_symbolic_signals,
            lookup_table=self.lookup_table_,
            _features_with_symbols_nonumreduc_noquantifseglen_df=_features_with_symbols_nonumreduc_noquantifseglen_df,
            _features_with_symbols_noquantifseglen_df=_features_with_symbols_noquantifseglen_df,
            _features_with_symbols_df=features_with_symbols_df,
        )
        return b_transform_symbolization

    def transform_quantif(self, segment_features_df: pd.DataFrame):
        """Return the segment symbols using quantification."""

        err_msg = "Run `.fit()` first."
        assert self.y_quantif_bins_ is not None, err_msg

        # Retrieve the features
        features = self.get_feat_df(segment_features_df=segment_features_df)

        # Get symbols
        segment_symbols = np.digitize(x=features, bins=self.y_quantif_bins_)
        return segment_symbols

    def transform_clustering(self, segment_features_df: pd.DataFrame):
        """Return the segment symbols using clustering."""

        err_msg = "Run `.fit()` first."
        assert self.scaler_ is not None, err_msg
        assert self.clustering_model_ is not None, err_msg

        # Retrieve and scale the features
        scaled_features = self.scaler_.transform(
            self.get_feat_df(segment_features_df=segment_features_df)
        )
        scaled_features_df = pd.DataFrame(
            scaled_features,
            columns=self.scaler_.feature_names_in_
        )
        # NEW
        if self.features_scaling is not None:
            scaled_features_df["length_feat"] = scaled_features_df["length_feat"] * \
                self.features_scaling["length_feat"]

        # Getting the cluster labels per segment
        segment_symbols = self.clustering_model_.predict(scaled_features_df)
        return segment_symbols

    def transform_quantif_seglen(self, segment_lengths: pd.Series):
        """Quantify a series of segment lengths and dividing them by their minimum.

        When `type(self.n_regime_lengths) == int`.
        """

        err_msg = "Run `.fit()` first."
        assert self.from_seglen_label_to_value_dict_ is not None, err_msg
        assert self.seglen_bins_ is not None, err_msg

        # Get the quantified segment lengths
        quantified_seglen = self.apply_quantif_seglen(
            segment_lengths=segment_lengths,
            seglen_bins=self.seglen_bins_,
            from_seglen_label_to_value_dict=self.from_seglen_label_to_value_dict_,
        )

        # Reduce the quantified segment lengths (to make the symbolic
        # signals shorter)
        scaled_quantified_seglen = self.shorten_quantif_seglen(
            quantified_seglen)

        return scaled_quantified_seglen

    @staticmethod
    def apply_quantif_seglen(
        segment_lengths: pd.Series,
        seglen_bins: list,
        from_seglen_label_to_value_dict: dict
    ):
        """Quantify a series of segment lengths (without dividing them by
        their minimum).
        """

        # Label each segment length
        labeled_segment_lengths = np.digitize(
            segment_lengths,
            bins=seglen_bins,
        )

        # Apply the quantification dictionary on the segment length labels
        quantified_seglen = np.vectorize(
            from_seglen_label_to_value_dict.get
        )(labeled_segment_lengths)

        return quantified_seglen

    @staticmethod
    def shorten_quantif_seglen(quantified_seglen: pd.Series):
        """Reduce the segment lengths (to make the symbolic
        sequences shorter)"""

        min_quantified_seglen = quantified_seglen.min()
        scaled_quantified_seglen = (
            quantified_seglen / min_quantified_seglen).round(0).astype(int)
        return scaled_quantified_seglen

    @staticmethod
    def compute_lookup_table_mindist(y_quantif_bins) -> np.ndarray:
        """
        Compute the lookup table which is called by the MINDIST function.
        """
        n_symbols = len(y_quantif_bins) + 1
        lookup_table = np.zeros((n_symbols, n_symbols))
        for i_row in range(n_symbols):
            for i_column in range(i_row + 2, n_symbols):
                lookup_table[i_row, i_column] = (
                    y_quantif_bins[i_column - 1] - y_quantif_bins[i_row]
                )
        lookup_table += lookup_table.T  # because the matrix is symmetric
        return lookup_table

    @staticmethod
    def compute_lookup_table_mof(segment_symbols, feature_1D) -> np.ndarray:
        """
        Compute the lookup table for the mean of {univariate feature
        per segment}.
        """
        df = pd.DataFrame(
            {
                "symbol": segment_symbols.flatten(),
                "feature": feature_1D.flatten(),
            }
        )
        # mean of feature (mof)
        mof = df.sort_values("symbol").groupby("symbol").mean().to_numpy()
        lookup_table = squareform(pdist(X=mof, metric="euclidean"))
        return lookup_table

    @staticmethod
    def transform_numerosity_reduction(features_with_symbols_df) -> pd.DataFrame:
        """"Apply numerosity reduction (fusion of segments)."""

        l_index_rows_allsignals = list()

        for (_, group) in features_with_symbols_df.groupby("signal_index"):

            # For a signal, get the rows where we should merge adjacent
            # segments because they have the same (redundant) symbol
            group["segment_symbol_diff"] = (
                group["segment_symbol"]+1).diff()
            l_index_rows_signal = group.index[group["segment_symbol_diff"] == 0].tolist(
            )
            l_index_rows_allsignals.append(l_index_rows_signal)

            # Update `segment_end` and `segment_length` for the segment to be
            # merged
            for index_row in sorted(l_index_rows_signal, reverse=True):
                features_with_symbols_df.loc[index_row-1,
                                             "segment_end"] = features_with_symbols_df["segment_end"].iloc[index_row]
                features_with_symbols_df.loc[index_row-1,
                                             "segment_length"] += features_with_symbols_df["segment_length"].iloc[index_row]

        l_index_rows_allsignals = sorted(
            list(np.concatenate(l_index_rows_allsignals).astype(int).flat))

        # Drop the segments who have been absorbed after the merge
        features_with_symbols_df = features_with_symbols_df.drop(
            l_index_rows_allsignals).reset_index(drop=True)

        # TODO: careful, the mean feature is no longer accurate
        # TODO: maybe update the look-up table

        return features_with_symbols_df
