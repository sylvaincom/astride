from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from tslearn.metrics.cysax import cydist_1d_sax, cydist_sax
from tslearn.piecewise import (OneD_SymbolicAggregateApproximation,
                               SymbolicAggregateApproximation)

from src.symbolic_signal_distance import SymbolicSignalDistance


class MySAX(SymbolicAggregateApproximation):
    def _transform(self, X, y=None, *args, **kwargs):
        return Bunch(
            data=super()._transform(X=X, y=y),
            breakpoints_avg_=self.breakpoints_avg_,
            _X_fit_dims_=self._X_fit_dims_
        )


class DistInterfaceSAX(BaseEstimator):
    def fit(self, X, y=None, *args, **kwargs):
        self.X_transformed = X.data
        self.breakpoints_avg_ = X.breakpoints_avg_
        self._X_fit_dims_ = X._X_fit_dims_
        return self

    def distance_sax(self, sax1, sax2):
        return cydist_sax(
            sax1,
            sax2,
            self.breakpoints_avg_,
            self._X_fit_dims_[1]
        )

    def transform(self, X, y=None):
        distance_matrix = SymbolicSignalDistance.collection_dist(
            list_of_test_signals=X.data,
            list_of_train_signals=self.X_transformed,
            metric=self.distance_sax
        )
        return distance_matrix  # (n_samples_test, n_samples_train)


class MyOneD(OneD_SymbolicAggregateApproximation):
    def _transform(self, X, y=None, *args, **kwargs):
        return Bunch(
            data=super()._transform(X=X, y=y),
            breakpoints_avg_middle_=self.breakpoints_avg_middle_,
            breakpoints_slope_middle_=self.breakpoints_slope_middle_,
            _X_fit_dims_=self._X_fit_dims_,
        )


class DistInterfaceOneD(BaseEstimator):
    def fit(self, X, y=None, *args, **kwargs):
        self.X_transformed = X.data
        self.breakpoints_avg_middle_ = X.breakpoints_avg_middle_
        self.breakpoints_slope_middle_ = X.breakpoints_slope_middle_
        self._X_fit_dims_ = X._X_fit_dims_
        return self

    def distance_1D_sax(self, sax1, sax2):
        return cydist_1d_sax(
            sax1,
            sax2,
            self.breakpoints_avg_middle_,
            self.breakpoints_slope_middle_,
            self._X_fit_dims_[1],
        )

    def transform(self, X, y=None):
        distance_matrix = SymbolicSignalDistance.collection_dist(
            list_of_test_signals=X.data,
            list_of_train_signals=self.X_transformed,
            metric=self.distance_1D_sax,
        )
        return distance_matrix  # (n_samples_test, n_samples_train)
