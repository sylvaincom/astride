import numpy as np
from sklearn.utils import Bunch
from weighted_levenshtein import lev
from sklearn.base import BaseEstimator

class SymbolicSignalDistance(BaseEstimator):
    """
    Computes the distance between symbolic signals.

    Inputs the symbolic signals and the look-up table which is the matrix of
        pairwise distances between individual symbols.
    Outputs the matrix of pairwise distances between symbolic signals.

    Parameters
    ----------
    distance : {'euclidean', 'lev'}, default='euclidean'
        Distance type for going from the distance between pairwise individual
            symbols (`lookup_table`) to the distance between pairwise symbolic
            signals. Possible values:

            - 'euclidean' : Euclidean distance type.
            - 'lev' : general Levenshtein distance (a.k.a. general edit
                distance), from the `weighted_levenshtein` Python library.

    n_samples : int, default=100
        Number of samples per (raw) input signal, which is the length of the
            input signals, assumed to be the same for all the signals in the
            same data set in `list_of_signals`.

    weighted_bool : bool, default=True
        Boolean deciding if we will use the computed `lookup_table` or not.
        Possible values:

        - False : the computed `lookup_table` will be ignored and the distance
            between individual symbols will be of 1 everywhere.
        - True : the distance between individual symbols will be given by the
            computed `lookup_table`. If distance='lev', then the distance
            between symbolic signals is the weighted Levenshtein distance.
    """

    def __init__(
        self,
        distance: str = "euclidean",
        n_samples: int = 100,
        weighted_bool: bool = True,
    ) -> None:

        # Unit tests on the parameters:

        err_msg = f"Choose an existing distance type, not {distance}."
        assert distance in ["euclidean", "lev"], err_msg

        if distance == "euclidean":
            err_msg = "Provide the number of samples per time series."
            assert n_samples is not None, err_msg

        # Initializing the parameters
        self.distance = distance
        self.n_samples = n_samples
        self.weighted_bool = weighted_bool

    def fit(self, b_transform_symbolization, *args, **kwargs):
        self.list_of_symbolic_signals_train = (
            b_transform_symbolization.list_of_symbolic_signals
        )
        if self.weighted_bool:
            self.lookup_table_ = b_transform_symbolization.lookup_table
        else:
            self.lookup_table_ = np.ones(
                b_transform_symbolization.lookup_table.shape
            )
        return self

    def transform(self, b_transform_symbolization):
        list_of_symbolic_signals_test = (
            b_transform_symbolization.list_of_symbolic_signals
        )
        if self.distance == "euclidean":
            distance_matrix = self.collection_dist(
                list_of_test_signals=list_of_symbolic_signals_test,
                list_of_train_signals=self.list_of_symbolic_signals_train,
                metric=self.compute_euclidean,
                lookup_table=self.lookup_table_,
            )
            return distance_matrix
        elif self.distance == "lev":
            # Transforming the costs:
            b_transform_costs = self.transform_costs(self.lookup_table_)
            # Computing the distance matrix:
            distance_matrix = self.collection_dist(
                list_of_test_signals=list_of_symbolic_signals_test,
                list_of_train_signals=self.list_of_symbolic_signals_train,
                metric=self.compute_weighted_lev,
                insert_costs=b_transform_costs.insert_costs,
                delete_costs=b_transform_costs.delete_costs,
                substitute_costs=b_transform_costs.substitute_costs,
            )
            return distance_matrix

    def compute_euclidean(self, symb_signal_1, symb_signal_2, lookup_table):
        """Compute the euclidean type distance between two symbolic signals.

        The distance is normalized by the lengths of the symbolic signals.
        """

        err_msg = (
            "If the distance is `euclidean`, the lengths of the "
            "symbolic signals must be the same. Otherwise, use `lev`."
        )
        word_length_1 = len(symb_signal_1)
        word_length_2 = len(symb_signal_2)
        assert word_length_1 == word_length_2, err_msg

        sum_cells = 0
        for (symbol_1, symbol_2) in zip(symb_signal_1, symb_signal_2):
            sum_cells += (lookup_table[symbol_1][symbol_2]) ** 2
        symb_signals_dist = np.sqrt(self.n_samples / word_length_1) * np.sqrt(
            sum_cells
        )

        return symb_signals_dist

    def compute_weighted_lev(
        self,
        symb_signal_1,
        symb_signal_2,
        insert_costs,
        delete_costs,
        substitute_costs,
    ):
        """Compute the general edit distance (a.k.a weighted Levenshtein
        distance) between two symbolic signals.

        The distance is not normalized by the lengths of the symbolic signals.
        symb_signal_1 and symb_signal_2 are signals of integers (the labels of
        the segment classes).
        """

        # Avoid weird ASCII characters
        n_symbols = len(self.lookup_table_)
        assert n_symbols <= 26, "`n_symbols` should be inferior to 26!"
        alphabet_signal_1 = [chr(i + ord("A")) for i in symb_signal_1]
        alphabet_signal_2 = [chr(i + ord("A")) for i in symb_signal_2]

        # Convert the list of strings / characters into long strings:
        str_alphabet_signal_1 = "".join(alphabet_signal_1)
        str_alphabet_signal_2 = "".join(alphabet_signal_2)

        # Compute the weighted Levenshtein distance:
        symb_signals_dist = lev(
            str_alphabet_signal_1,
            str_alphabet_signal_2,
            insert_costs=insert_costs,
            delete_costs=delete_costs,
            substitute_costs=substitute_costs,
        )
        return symb_signals_dist

    @staticmethod
    def transform_costs(lookup_table):
        """Transform the substitute, insertion and deletion costs.

        Computed from the look-up table and used for the weighted Levenshtein
        distance.

        Our symbols are the A, B, C, ... ASCII characters.
        """

        # Integrate the lookup table into the substitute costs:
        substitute_costs = np.ones((128, 128), dtype=np.float64)
        n_symbols = lookup_table.shape[0]
        substitute_costs[
            ord("A"): ord("A") + n_symbols, ord("A"): ord("A") + n_symbols
        ] = lookup_table.astype(np.float64)

        # Scale up the insert and delete costs:
        lookup_table_max = lookup_table.max()
        insert_costs = np.ones(128, dtype=np.float64) * lookup_table_max
        delete_costs = np.ones(128, dtype=np.float64) * lookup_table_max

        b_transform_costs = Bunch(
            insert_costs=insert_costs,
            delete_costs=delete_costs,
            substitute_costs=substitute_costs,
        )
        return b_transform_costs

    @staticmethod
    def pairwise_dist(list_of_signals, metric, *args, **kwargs) -> np.ndarray:
        """Compute the pairwise distances defined by the function `metric`.

        The arguments *args and **kwargs are passed to the metric function.

        TODO: If not used by our class, remove this function.
        """
        n_signals = len(list_of_signals)
        distance_matrix = np.zeros((n_signals, n_signals), dtype=float)
        for (i_row, signal_1) in enumerate(list_of_signals):
            for (i_column, signal_2) in enumerate(
                list_of_signals[i_row + 1:], start=i_row + 1
            ):
                distance_matrix[i_row, i_column] = metric(
                    signal_1, signal_2, *args, **kwargs
                )
        distance_matrix += distance_matrix.T
        return distance_matrix

    @staticmethod
    def collection_dist(
        list_of_test_signals, list_of_train_signals, metric, *args, **kwargs
    ) -> np.ndarray:
        """Compute distance between each pair of the two collections of inputs
        defined by the function `metric`.

        The arguments *args and **kwargs are passed to the metric function.
        """
        n_test_signals = len(list_of_test_signals)
        n_train_signals = len(list_of_train_signals)
        distance_matrix = np.zeros(
            (n_test_signals, n_train_signals), dtype=float
        )
        for (i_row, test_signal) in enumerate(list_of_test_signals):
            for (i_column, train_signal) in enumerate(list_of_train_signals):
                distance_matrix[i_row, i_column] = metric(
                    test_signal, train_signal, *args, **kwargs
                )
        return distance_matrix
