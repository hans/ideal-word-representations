"""
ERP/epoching visualizations
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm


def plot_epoch_raster(epochs_df, electrode_df, order=None, sort_epochs=True):
    raster_df = epochs_df.pivot(index=["electrode_idx", "epoch_idx"],
                                columns="epoch_time",
                                values="value")
    if order is None:
        order = sorted(raster_df.index.levels[0].unique())
    
    num_rows = len(raster_df.index.levels[0])
    
    f, axs = plt.subplots(num_rows, 1, figsize=(14, 6 * num_rows))
    vmin = raster_df.min().min()
    vmax = raster_df.max().max()
    for ax, electrode_idx in zip(tqdm(axs), order):
        electrode_epochs = raster_df.loc[electrode_idx]

        if sort_epochs:
            electrode_epochs = electrode_epochs.loc[electrode_epochs.idxmax(axis=1).sort_values(ascending=False).index]

        sns.heatmap(electrode_epochs, ax=ax, cmap="RdBu", center=0, cbar=True, vmin=vmin, vmax=vmax)
        # ax.imshow(electrode_epochs, aspect="auto", cmap="viridis")
        ax.set_title(f"{electrode_idx} // {electrode_df.loc[electrode_idx].electrode_name}")

        # ax.set_xticklabels([electrode_epochs.columns[int(idx.get_text())] for idx in ax.get_xticklabels()])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Epoch idx")