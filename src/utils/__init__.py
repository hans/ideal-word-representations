import numpy as np
import pandas as pd


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