from pathlib import Path
import re
from typing import Union, Callable

import numpy as np
import pandas as pd


def concat_csv_with_indices(path_glob: str,
                            path_patterns: list[Union[str, re.Pattern, Callable[[Path], str]]],
                            index_names: list[str]):
    """
    Load a collection of dataframes organized within folders, and use patterns on the
    folder names to create a concatenated multi-indexed dataframe.

    Args:
    path_glob: a glob pattern for the folders containing the dataframes
    path_patterns: a list of patterns to apply to the path names to extract indices
    index_names: a list of names for the indices
    """
    assert len(path_patterns) == len(index_names)

    paths = list(Path().glob(path_glob))
    dfs = [pd.read_csv(p) for p in paths]
    index_keys = [
        tuple([patt(p) if callable(patt) else re.search(patt, str(p)).group(1)
               for patt in path_patterns])
        for p in paths
    ]

    return pd.concat(dfs, keys=index_keys, names=index_names)


def ndarray_to_long_dataframe(ndarray: np.ndarray, axis_names: list[str]):
    """
    Convert a multi-dimensional NumPy ndarray to a long-format pandas DataFrame.
    
    Parameters:
    - ndarray: The multi-dimensional NumPy ndarray to convert.
    - axis_names: A list of names for each axis of the ndarray.
    
    Returns:
    - A pandas DataFrame in long format with a MultiIndex.
    """
    # Check if the number of axis names matches the number of dimensions
    if len(axis_names) != ndarray.ndim:
        raise ValueError("Number of axis names must match the number of dimensions in the ndarray")
    
    # Get the shape of the ndarray
    shape = ndarray.shape
    
    # Create a list of ranges for each dimension
    ranges = [range(dim) for dim in shape]
    
    # Generate a grid of coordinates for each element in the ndarray
    coords = np.array(np.meshgrid(*ranges, indexing='ij')).reshape(ndarray.ndim, -1).T
    
    # Flatten the ndarray
    flattened_values = ndarray.flatten()
    
    # Create a MultiIndex from the coordinates
    multi_index = pd.MultiIndex.from_arrays(coords.T, names=axis_names)
    
    # Create a DataFrame with the MultiIndex and the flattened values
    df = pd.DataFrame(flattened_values, index=multi_index, columns=['value'])
    
    return df