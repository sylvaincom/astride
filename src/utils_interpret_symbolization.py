from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import create_path

cwd = Path.cwd()
plt.rcParams['figure.figsize'] = (10, 3)
sns.set_theme()


def plot_multivariate_segmentation(
    b_dataset,
    b_transform_method_dataset,
    is_save_fig=False,
    method_name="unkown",
    date_exp="unknown",
):
    dataset_name_ucr = b_dataset.dataset_name_ucr
        
    # Get the change-points of the segmentation (that are common to all signals)
    bkps = b_transform_method_dataset.list_of_bkps[b_dataset.signal_indexes[0]]
    bkps.append(0)
    bkps = sorted(bkps)
    bkps[-1] = bkps[-1]-1
    
    # Create the figure
    fig, axs = plt.subplots(
        len(b_dataset.signal_indexes), 1, figsize=(8, 6),
        constrained_layout=True, sharex=True, sharey=True
    )
    
    # For each signal, plot the original signal and its change-points
    for (i_axis, signal_index) in enumerate(b_dataset.signal_indexes):
        normalized_signal = b_transform_method_dataset.list_of_scaled_signals[signal_index]
        signal_label = b_dataset.y_train[signal_index]
        
        axs[i_axis].plot(normalized_signal, 'g-', linewidth=2, label='normalized signal')
        
        # Plot the vertical segmentation bins
        for (i, bkp) in enumerate(bkps):
            if i==0:
                axs[i_axis].axvline(
                    x=bkp, color='b', linestyle='--', label='segmentation bins', alpha=0.7)
            else:
                axs[i_axis].axvline(
                    x=bkp, color='b', linestyle='--', alpha=0.7)
        
        axs[i_axis].legend(fontsize=10, loc="upper left")
        axs[i_axis].set_title(
            f"Signal index: {signal_index}. Class label: {signal_label}.", fontsize=10, loc="left")
    plt.tight_layout()
    plt.margins(x=0)
    if is_save_fig:
        folder = cwd / f"results/{date_exp}/img"
        create_path(folder)
        plt.savefig(folder / f"multivariate_change_points_{method_name}_{dataset_name_ucr}.png", dpi=200)
    plt.show()


def plot_symbolization(
    b_dataset,
    b_transform_method_dataset,
    features_with_symbols_label_df_method_dataset,
    is_save_fig=False,
    method_name="unkown",
    date_exp="unknown",
):
    dataset_name_ucr = b_dataset.dataset_name_ucr
    n_symbols = b_dataset.n_symbols
    
    # Get the (original) normalized signal
    signal = b_transform_method_dataset.list_of_scaled_signals[b_dataset.signal_index]
    
    # Get the change-points of the segmentation (that are common to all signals)
    bkps = b_transform_method_dataset.list_of_bkps[b_dataset.signal_index]
    bkps.append(0)
    bkps = sorted(bkps)
    bkps[-1] = bkps[-1]-1
    
    # Create the figure
    plt.figure(figsize=(18*0.8, 6*0.7))
    
    # Plot the original signal
    plt.plot(signal, 'go-', label='normalized signal', alpha=0.5)

    # Plot the vertical segmentation bins
    for (i, bkp) in enumerate(bkps):
        if i==0:
            plt.axvline(
                x=bkp, color='b', linestyle='--', label='segmentation bins', alpha=0.7)
        else:
            plt.axvline(
                x=bkp, color='b', linestyle='--', alpha=0.7)
    
    # Plot the mean per segment representation
    df_signal = features_with_symbols_label_df_method_dataset.query(f"signal_index == {b_dataset.signal_index}")
    mean_per_segment_representation = np.array(
        df_signal.mean_feat.apply(lambda x: [x])
        * df_signal.segment_length.astype(int)
    ).sum()
    plt.plot(mean_per_segment_representation, "b-", alpha=0.7, label="mean per segment")
    
    # Plot the horizontal quantizations bins
    y_quantif_bins = b_transform_method_dataset.y_quantif_bins
    if y_quantif_bins is not None:
        for (i, y_quantif_bin) in enumerate(y_quantif_bins):
            if i==0:
                plt.axhline(
                    y=y_quantif_bin,
                    color="r",
                    linestyle="dashed",
                    alpha=0.7,
                    label="quantization bins",
                )
            else:
                plt.axhline(y=y_quantif_bin, color="r",
                            linestyle="dashed", alpha=0.7)
        
    # Add the symbols corresponding to each quantization bin:
    if y_quantif_bins is not None:
        for symbol in range(n_symbols):
            if symbol == 0:
                y = np.mean([signal.min(), y_quantif_bins[symbol]])
                plt.text(
                    0,
                    y,
                    symbol,
                    fontsize=18,
                    color="r",
                    style="italic",
                    bbox={"facecolor": "lightgrey", "alpha": 0.7, "pad": 5},
                    horizontalalignment="left",
                )
            elif symbol == n_symbols - 1:
                y = np.mean([y_quantif_bins[-1], signal.max()])
                plt.text(
                    0,
                    y,
                    n_symbols-1,
                    fontsize=18,
                    color="r",
                    style="italic",
                    bbox={"facecolor": "lightgrey", "alpha": 0.7, "pad": 5},
                    horizontalalignment="left",
                )
            else:
                y = np.mean([y_quantif_bins[symbol - 1],
                            y_quantif_bins[symbol]])
                plt.text(
                    0,
                    y,
                    symbol,
                    fontsize=18,
                    color="r",
                    style="italic",
                    bbox={"facecolor": "lightgrey", "alpha": 0.7, "pad": 5},
                    horizontalalignment="left",
                )
    
    # Display the symbolic sequence
    for i in range(len(features_with_symbols_label_df_method_dataset)):
        x = features_with_symbols_label_df_method_dataset["segment_start"].iloc[i]
        y = features_with_symbols_label_df_method_dataset["mean_feat"].iloc[i]
        h = features_with_symbols_label_df_method_dataset["segment_length"].iloc[i]
        symbol = features_with_symbols_label_df_method_dataset["segment_symbol"].iloc[i]
        plt.text(
            x + h / 2,
            y,
            symbol,
            fontsize=18,
            color="r",
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
        )

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.margins(x=0)
    if is_save_fig:
        folder = cwd / f"results/{date_exp}/img"
        create_path(folder)
        plt.savefig(folder / f"symbolization_{method_name}_{dataset_name_ucr}_{b_dataset.signal_index}.png", dpi=200)
    plt.show()