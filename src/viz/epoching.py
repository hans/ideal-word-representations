"""
ERP/epoching visualizations
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm


def plot_epoch_raster(epochs_df, electrode_df, electrode_order=None,
                      epoch_order: Optional[np.ndarray] = None):
    raster_df = epochs_df.pivot(index=["electrode_idx", "epoch_idx"],
                                columns="epoch_time",
                                values="value")
    
    if electrode_order is None:
        electrode_order = sorted(raster_df.index.levels[0].unique())

    if epoch_order is None:
        epoch_order = np.array(sorted(raster_df.index.levels[1].unique()))
    if epoch_order.ndim == 1:
        epoch_order = np.tile(epoch_order[None, :], (len(electrode_order), 1))
    
    num_rows = len(raster_df.index.levels[0])
    
    f, axs = plt.subplots(num_rows, 1, figsize=(14, 6 * num_rows))
    if num_rows == 1:
        axs = [axs]

    vmin = raster_df.min().min()
    vmax = raster_df.max().max()
    for ax, electrode_idx, epoch_order_i in zip(tqdm(axs), electrode_order, epoch_order):
        electrode_epochs = raster_df.loc[electrode_idx].loc[epoch_order_i]

        # if sort_epochs:
        #     electrode_epochs = electrode_epochs.loc[electrode_epochs.idxmax(axis=1).sort_values(ascending=False).index]

        sns.heatmap(electrode_epochs, ax=ax, cmap="RdBu", center=0, cbar=True, vmin=vmin, vmax=vmax)
        # ax.imshow(electrode_epochs, aspect="auto", cmap="viridis")
        ax.set_title(f"{electrode_idx} // {electrode_df.loc[electrode_idx].electrode_name}")

        # ax.set_xticklabels([electrode_epochs.columns[int(idx.get_text())] for idx in ax.get_xticklabels()])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Epoch idx")

    return axs