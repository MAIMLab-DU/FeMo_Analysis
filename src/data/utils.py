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
    """
    author: @mnakash
    
    Perform binary dilation on input binary data using a dilate with binary structure of size dilation_size.
    Dilation will be performed in both direction with half of the dilation_size.
    
    Example:
        Input   data = np.array([0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,0])
                dilation_size = 3
        Output  dilated_data  = np.array([0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0])
        
    Parameters:
    - data: Binary input data (1D NumPy array)
    - dilation_size: Binary structuring element size (int)

    Returns:
    - dilated_data (1D NumPy array)
    """    
    
    labelled = label(data)  # get number of spikes labelling them in ascending order
    interested_index = []  # to hold indices where to perform dilation
    
    
    for i in range(1, max(labelled)+1):  # from 1 to total no_of_spikes
        indexs = np.where(labelled == i)[0]  # indexs where i is found
        if len(indexs)>1:
            # store only firts and last index
            interested_index.append(indexs[0])
            interested_index.append(indexs[-1])
        else:
            interested_index.append(indexs[0])
    
    
    dilate_before = dilation_size//2
    dilate_after = dilation_size-dilate_before -1  # minus 1 to exclude current index
    
    dilated_data = data.copy()
    for i in interested_index:
        start_idx = max(0, i - dilate_before)
        end_idx = min(i + dilate_after, len(data)-1)
        
        dilated_data[start_idx:end_idx+1] = 1

    return dilated_data
