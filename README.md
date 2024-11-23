# ASTRIDE: Adaptive Symbolization for Time Series Databases

This repository contains the code to reproduce all experiments in our preprint paper
>ASTRIDE: Adaptive Symbolization for Time Series Databases
available on [arXiv](https://arxiv.org/abs/2302.04097).

All the code is written in Python (scripts and notebooks).

Note that the ASTRIDE method has been accepted at [EUSIPCO 2024](https://ieeexplore.ieee.org/document/10715214):
> S. W. Combettes, C. Truong, and L. Oudre. "Symbolic representation for time series." In _Proceedings of the European Signal Processing Conference (EUSIPCO)_, Lyon, France, 2024.

<details><summary><i>Toggle for the paper's abstract!</i></summary>We introduce ASTRIDE (Adaptive Symbolization for Time seRIes DatabasEs), a novel symbolic representation of time series, along with its accelerated variant FASTRIDE (Fast ASTRIDE). Unlike most symbolization procedures, ASTRIDE is adaptive during both the segmentation step by performing change-point detection and the quantization step by using quantiles. Instead of proceeding signal by signal, ASTRIDE builds a dictionary of symbols that is common to all signals in a data set. We also introduce D-GED (Dynamic General Edit Distance), a novel similarity measure on symbolic representations based on the general edit distance. We demonstrate the performance of the ASTRIDE and FASTRIDE representations compared to SAX (Symbolic Aggregate approXimation), 1d-SAX, SFA (Symbolic Fourier Approximation), and ABBA (Adaptive Brownian Bridge-based Aggregation) on reconstruction and, when applicable, on classification tasks. These algorithms are evaluated on 86 univariate equal-size data sets from the UCR Time Series Classification Archive. An open source GitHub repository called astride is made available to reproduce all the experiments in Python.</details></br>

Please let us know of any issue you might encounter when using this code, either by opening an issue on this repository or by sending an email to `sylvain.combettes8 [at] gmail.com`. Or if you just need some clarification or help.

## How is a symbolic representation implemented?

For ASTRIDE and FASTRIDE, a symbolic representation (with an associated distance) is a scikit-learn pipeline based on the following classes in the `src` folder:
1. `SegmentFeature` (in `segmentation.py`)
1. `Segmentation` (in `segment_features.py`)
1. `Symbolization` (in `symbolization.py`)
1. `SymbolicSignalDistance` (in `symbolic_signal_distance.py`)

An example usage is given in `07_run_reconstruction.py`.

Download the `ABBA/ABBA.py` file from https://github.com/nla-group/ABBA, as it is used in the signal reconstruction benchmark.

## Structure of the code

`date_exp` is a string (for example `"2023_02_08"`) in order to version the experiments.

The code outputs the following kinds of `csv` files:
1. `data` folder: the UCR archive meta data filtered on the list of data sets under consideration
1. `results/{date_exp}/acc` folder: the results from the 1-NN classifcation task
1. `results/{date_exp}/clean_acc` folder: the results from the 1-NN classifcation task
1. `results/{date_exp}/reconstruction` folder: the results from signal reconstruction task

The code outputs all the figures in the paper in the `results/{date_exp}/img` folder.

For the Python notebooks, the configuration parameters (at the top) do the following:
- `DATE_EXP` sets the date of launch of the experiment (for versioning)
- if `IS_EXPORT_DF=False`, then no CSV files (pandas DataFrame) are exported
- if `IS_SAVE_FIG=False`, then no figures are exported
- if `IS_COMPUTE=False`, then long computations are made (no more than a few mintutes on a standard laptop)

## How to use this repository to reproduce the ASTRIDE paper

_Note that more details are given at the top of each notebook and that the code is commented._

1. Explore the UCR archive.<br>
    Run the `01_explore_ucr_metadata.ipynb` notebook.<br>
    It generates:
    - the `data/DataSummary_prep_equalsize.csv` file which contains the 117 univariate and equal-size data sets from the UCR archive.
    - the `data/DataSummary_prep_equalsize_min100samples.csv` file which contains the 94 univariate and equal-size data sets with at least 100 samples from the UCR archive.
    - Table 4.
    - Data for Section 2.3.3 and Section 3.4 about the total memory usage of a symbolization method.
1. Look into the normality assumption oof the means per segment.<br>
    Run the `02_explore_gaussian_assumption.ipynb` notebook.<br>
    - It performs the normality test for section 2.3.1.
    - It generates Figure 2.
1. Intrepret the symbolization methods, how they transform a signal.<br>
    Run the `03_interpret_symbolization.ipynb` notebook
    - It generates Figures 1, 3 and 4.
    - It generates Table 3.
1. For the classification benchmark of SAX, 1d-SAX, ASTRIDE, and FASTRIDE of Section 4.1.
    1. Compute the test accuracies of each method.<br>
        Run the `04_run_classification.py` script for the four symbolization methods:
        ```
        $ python3 04_run_classification.py --method_name "saxtslearn" --date_exp "2023_02_08"
        $ python3 04_run_classification.py --method_name "1dsax" --date_exp "2023_02_08"
        $ python3 04_run_classification.py --method_name "fastride" --date_exp "2023_02_08"
        $ python3 04_run_classification.py --method_name "astride" --date_exp "2023_02_08"
        ```
        Note that the `date_exp` variable is for versioning the experiments.<br>
        It generates the `results/{date_exp}/acc/df_acc_{method}_{dataset}.csv` files, which are the test accuracies per method per data set and for all combination of hyper-parameters.
    1. Clean the classification results for each method.<br>
        Run the `05_clean_classification_results.ipynb` notebook.
        It generates the `results/{date_exp}/acc_clean/df_acc_{method}_alldatasets_clean.csv` files, which are the test accuracies per method for all data sets and combination of hyper-parameters, cleaned.
    1. Explore the results for each method: plot the accuracy as a function of the word length.<br>
        Run the `06_explore_classification_results.ipynb` notebook.<br>
        It generates Figure 5.
1. For the signal reconstruction benchmark of SAX, 1d-SAX, SFA, ABBA, ASTRIDE and FASTRIDE of Section 4.2.
    1. Compute the reconstructed signals for each method and all signals of all data sets as well as the reconstruction error.<br>
        Run the `07_run_recontruction.py` script for several target memory usage ratios
        ```
        python3 07_run_reconstruction.py --denom 3 --date_exp "2023_02_08"
        python3 07_run_reconstruction.py --denom 4 --date_exp "2023_02_08"
        python3 07_run_reconstruction.py --denom 5 --date_exp "2023_02_08"
        python3 07_run_reconstruction.py --denom 6 --date_exp "2023_02_08"
        python3 07_run_reconstruction.py --denom 10 --date_exp "2023_02_08"
        python3 07_run_reconstruction.py --denom 15 --date_exp "2023_02_08"
        python3 07_run_reconstruction.py --denom 20 --date_exp "2023_02_08"
        ```
        Note that the `denom` variable is the inverse of the target memory usage ratio.<br>
        It generates:
        - the `results/reconstruction/{denom}/reconstruction_errors_{dataset}.csv` files, which are the
        - the `results/reconstruction/{denom}/reconstructed_signals/reconstructed_{dataset}_{method}.csv` files, which are the reconstructed signals for each symbolization method.
    1. Compute the reconstruction errors.<br>
        Run the `08_explore_recontruction_results.ipynb` notebook.<br>
        It generates Figures 6 and 7.
    1. Interpret the signal reconstruction for each symbolization method.<br>
        Run the `09_interpret_reconstruction.ipynb` notebook.<br>
        It generates Figure 8.
1. Compare the symbolization and classification durations of SAX, 1d-SAX, ASTRIDE, and FASTRIDE.<br>
    Run the `10_processing_time.ipynb` notebook.<br>
    It generates Table 6.

## Requirements

- loadmydata==0.0.9
- matplotlib==3.4.1
- numpy==1.20.2
- pandas==1.2.4
- plotly==5.5.0
- ruptures==1.1.6.dev3+g1def4ff
- scikit-learn==1.0.1
- scipy==1.6.2
- seaborn==0.11.1
- statsmodels==0.13.1
- tslearn==0.5.2
- weighted-levenshtein==0.2.1

## Citing

If you use this code or publication, please cite (arXiv: https://arxiv.org/abs/2302.04097):
```bibtex
@article{2023_combettes_astride,
    doi = {10.48550/ARXIV.2302.04097},
    url = {https://arxiv.org/abs/2302.04097},
    author = {Combettes, Sylvain W. and Truong, Charles and Oudre, Laurent},
    title = {ASTRIDE: Adaptive Symbolization for Time Series Databases},
    journal = {arXiv preprint arXiv:2302.04097},
    year = {2023},
}
```

```bibtex
@INPROCEEDINGS{10715214,
  author={Combettes, Sylvain W. and Truong, Charles and Oudre, Laurent},
  booktitle={2024 32nd European Signal Processing Conference (EUSIPCO)}, 
  title={Symbolic Representation for Time Series}, 
  year={2024},
  pages={1962-1966},
  doi={10.23919/EUSIPCO63174.2024.10715214}}
```

## Licence

This project is licensed under the MIT License, see the `LICENSE.md` file for more information.

## Contributors

* [Sylvain W. Combettes](https://sylvaincom.github.io/) (Centre Borelli, ENS Paris-Saclay)
* [Charles Truong](https://charles.doffy.net/) (Centre Borelli, ENS Paris-Saclay)
* [Laurent Oudre](http://www.laurentoudre.fr/) (Centre Borelli, ENS Paris-Saclay)

## Acknowledgments

Sylvain W. Combettes is supported by the IDAML chair (ENS Paris-Saclay) and UDOPIA (ANR-20-THIA-0013-01).
Charles Truong is funded by the PhLAMES chair (ENS Paris-Saclay).
Part of the computations has been executed on Atos Edge computer, funded by the IDAML chair (ENS Paris-Saclay).

<p align="center">
<img width="700" src="https://github.com/boniolp/dsymb-playground/blob/main/figures/cebo_logos.png"/>
</p>