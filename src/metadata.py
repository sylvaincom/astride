from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from src.segment_feature import SegmentFeature
from src.segmentation import Segmentation
from src.symbolic_signal_distance import SymbolicSignalDistance
from src.symbolization import Symbolization

# 86 UCR data sets that are considered in the classification benchmark
l_datasets_classif_bench = [
    'Adiac',
    'ArrowHead',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'Car',
    'CBF',
    'ChlorineConcentration',
    'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'DiatomSizeReduction',
    'Earthquakes',
    'ECG5000',
    'ECGFiveDays',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'GunPoint',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'InlineSkate',
    'InsectWingbeatSound',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'OliveOil',
    'OSULeaf',
    'Phoneme',
    'Plane',
    'RefrigerationDevices',
    'ScreenType',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoPatterns',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga',
    'ACSF1',
    'BME',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'EthanolLevel',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Fungi',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'HouseTwenty',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PowerCons',
    'Rock'
]

# Method name corresponding to the abbreviation
D_REPLACE_METHOD_NAMES = {
    "sax":"SAX",
    "saxtslearn":"SAX (tslearn)",
    "1dsax":"1d-SAX",
    "astride":"ASTRIDE",
    "fastride":"FASTRIDE",
}

# SAX pipeline including scaling and 1-NN classification
pipe_sax = (
    make_pipeline(
        TimeSeriesScalerMeanVariance(),
        Segmentation(
            univariate_or_multivariate="multivariate",
            uniform_or_adaptive="uniform",
            mean_or_slope=None,
            n_segments=8,  # to be set
            pen_factor=None
        ),
        SegmentFeature(
            features_names=["mean"]
        ),
        Symbolization(
            n_symbols=5,  # to be set
            symb_method="quantif",
            symb_quantif_method="gaussian",
            symb_cluster_method=None,
            features_scaling=None,
            reconstruct_bool=False,
            n_regime_lengths=None,
            seglen_bins_method=None,
            lookup_table_type="mindist"
        ),
        SymbolicSignalDistance(
            distance="euclidean",
            n_samples=100,  # to be set
            weighted_bool=True
        ),
        KNeighborsClassifier(
            n_neighbors=1,
            metric="precomputed"
        )
    )
)

# ASTRIDE pipeline including scaling and 1-NN classification
pipe_astride = (
    make_pipeline(
        TimeSeriesScalerMeanVariance(),
        Segmentation(
            univariate_or_multivariate="multivariate",
            uniform_or_adaptive="adaptive",
            mean_or_slope="mean",
            n_segments=8,  # to be set
            pen_factor=None
        ),
        SegmentFeature(
            features_names=["mean"]
        ),
        Symbolization(
            n_symbols=5,  # to be set
            symb_method="quantif",
            symb_quantif_method="quantiles",
            symb_cluster_method=None,
            features_scaling=None,
            reconstruct_bool=True,
            n_regime_lengths="divide_exact",
            seglen_bins_method=None,
            lookup_table_type="mof"
        ),
        SymbolicSignalDistance(
            distance="lev",
            n_samples=None,
            weighted_bool=True
        ),
        KNeighborsClassifier(
            n_neighbors=1,
            metric="precomputed"
        )
    )
)

# FASTRIDE pipeline including scaling and 1-NN classification
pipe_fastride = (
    make_pipeline(
        TimeSeriesScalerMeanVariance(),
        Segmentation(
            univariate_or_multivariate="multivariate",
            uniform_or_adaptive="uniform",
            mean_or_slope=None,
            n_segments=8,  # to be set
            pen_factor=None
        ),
        SegmentFeature(
            features_names=["mean"]
        ),
        Symbolization(
            n_symbols=5,  # to be set
            symb_method="quantif",
            symb_quantif_method="quantiles",
            symb_cluster_method=None,
            features_scaling=None,
            reconstruct_bool=False,
            n_regime_lengths=None,
            seglen_bins_method=None,
            lookup_table_type="mof"
        ),
        SymbolicSignalDistance(
            distance="lev",
            n_samples=None,
            weighted_bool=True
        ),
        KNeighborsClassifier(
            n_neighbors=1,
            metric="precomputed"
        )
    )
)
