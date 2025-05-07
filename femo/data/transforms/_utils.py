import numpy as np
from skimage.measure import label


def apply_pca(df):
    # Implementing pca on IMU_roation_fltd to reduce the three dimensional values into one 
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(df)
    principal_component = principal_component.reshape(-1)
    return principal_component


def custom_binary_dilation(data, dilation_size):   
    
    # get number of spikes labelling them in ascending order
    labelled = label(data)
    # to hold indices where to perform dilation
    interested_index = []
    
    # from 1 to total no_of_spikes
    for i in range(1, max(labelled) + 1):
        # indexs where i is found
        indexs = np.where(labelled == i)[0]
        if len(indexs)>1:
            # store only firts and last index
            interested_index.append(indexs[0])
            interested_index.append(indexs[-1])
        else:
            interested_index.append(indexs[0])
    
    
    dilate_before = dilation_size//2
    # minus 1 to exclude current index
    dilate_after = dilation_size-dilate_before - 1
    
    dilated_data = data.copy()
    for i in interested_index:
        start_idx = max(0, i - dilate_before)
        end_idx = min(i + dilate_after, len(data) - 1)
        
        dilated_data[start_idx:end_idx+1] = 1

    return dilated_data


def custom_binary_erosion(data, erosion_size):
    """
    Perform binary erosion on input binary data using a binary structure of size erosion_size.
    Erosion will shrink the binary spikes (connected components of 1s) by half of the erosion_size
    from both directions.

    Example:
        Input   data = np.array([0, 1, 1, 1, 0, 1, 1, 0])
                erosion_size = 3
        Output  eroded_data = np.array([0, 0, 1, 0, 0, 1, 0, 0])

    Parameters:
    - data: Binary input data (1D NumPy array)
    - erosion_size: Binary structuring element size (int)

    Returns:
    - eroded_data (1D NumPy array)
    """
    labelled = label(data)  # Label the connected components
    eroded_data = data.copy()  # Create a copy of the input data

    erode_before = erosion_size // 2
    erode_after = erosion_size - erode_before - 1  # Minus 1 to exclude the current index

    for i in range(1, max(labelled) + 1):  # Loop through each labeled component
        indexs = np.where(labelled == i)[0]  # Indices of the current component
        if len(indexs) > erosion_size:
            # Shrink the component by keeping only the central part
            start_idx = indexs[0] + erode_before
            end_idx = indexs[-1] - erode_after
            eroded_data[indexs[0]:indexs[-1] + 1] = 0  # Clear the entire component
            eroded_data[start_idx:end_idx + 1] = 1  # Restore the eroded part
        else:
            # Remove the entire component if its size is less than erosion_size
            eroded_data[indexs] = 0

    return eroded_data


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "1"}:
        return True
    elif value.lower() in {"false", "no", "0"}:
        return False
    else:
        raise ValueError(f"Invalid boolean string: {value}")
