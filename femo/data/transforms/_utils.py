import struct
import numpy as np
from skimage.measure import label
from datetime import datetime, timezone


def apply_pca(df):
    """Apply Principal Component Analysis to reduce dimensionality to 1 component.
    
    Performs PCA on the input data to reduce three-dimensional values into a single
    dimensional representation, specifically designed for IMU rotation filtered data.
    
    Args:
        df (array-like): Input data array or DataFrame containing the multi-dimensional
            data to be reduced. Expected to have multiple columns/features.
    
    Returns:
        numpy.ndarray: 1D array containing the first principal component values,
            representing the reduced dimensional data.
    
    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 3)  # 3D data
        >>> reduced_data = apply_pca(data)
        >>> print(reduced_data.shape)  # (100,)
    """
    # Implementing pca on IMU_roation_fltd to reduce the three dimensional values into one 
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(df)
    principal_component = principal_component.reshape(-1)
    return principal_component


def custom_binary_dilation(data, dilation_size):   
    """Performs custom binary dilation on labeled spike data.
    This function identifies connected components (spikes) in binary data and dilates
    around the boundary points (first and last indices) of each spike region. For
    single-point spikes, dilation occurs around that single point.
    Args:
        data (array-like): Binary input data containing spike regions to be dilated.
        dilation_size (int): Size of the dilation kernel. The dilation extends
            dilation_size//2 points before and the remaining points after each
            boundary point.
    Returns:
        numpy.ndarray: Binary array with dilated spike regions, where dilated
            areas are set to 1.
    Note:
        The function only dilates around the first and last indices of each
        connected component, not the entire spike region. For single-point
        spikes, dilation occurs around that single point.
    """
    
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
    """Performs custom binary erosion on 1D labeled data.
    
    This function applies erosion to connected components in binary data by either
    shrinking components from both ends or removing them entirely if they're too small.
    
    Args:
        data: A 1D binary array containing the data to be eroded.
        erosion_size (int): The minimum size threshold for erosion. Components smaller
            than this size are completely removed, while larger components are shrunk
            by removing pixels from both ends.
            
    Returns:
        numpy.ndarray: A copy of the input data with erosion applied. Components
            smaller than erosion_size are set to 0, while larger components have
            their boundary pixels removed symmetrically from both ends.
            
    Note:
        The erosion is applied asymmetrically when erosion_size is even, with
        slightly more erosion applied to the beginning of each component.
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
    """Convert string representation of truth to boolean value.

    This function converts various string representations of boolean values
    to their corresponding boolean type. It handles common string representations
    like 'true', 'yes', '1' for True and 'false', 'no', '0' for False.

    Args:
        value: A string or boolean value to convert. Case-insensitive for strings.

    Returns:
        bool: The boolean representation of the input value.

    Raises:
        ValueError: If the string value cannot be converted to a boolean.

    Examples:
        >>> str2bool("true")
        True
        >>> str2bool("FALSE")
        False
        >>> str2bool("yes")
        True
        >>> str2bool(True)
        True
        >>> str2bool("maybe")
        ValueError: Invalid boolean string: maybe
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "1"}:
        return True
    elif value.lower() in {"false", "no", "0"}:
        return False
    else:
        raise ValueError(f"Invalid boolean string: {value}")


def timestamp_to_iso(timestamp_ms: int):
    """
    Convert a timestamp in milliseconds to ISO 8601 format string.

    Args:
        timestamp_ms (int): Timestamp in milliseconds since Unix epoch.

    Returns:
        str: ISO 8601 formatted datetime string in UTC timezone with millisecond 
             precision, ending with 'Z' to indicate UTC (e.g., '2023-12-25T10:30:45.123Z').

    Example:
        >>> timestamp_to_iso(1703505045123)
        '2023-12-25T10:30:45.123Z'
    """
    iso_utc_ms = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return iso_utc_ms


def check_file_format(filename: str) -> str:
    """
    Check if the first 4 bytes represent a Unix timestamp (V1 format) or magic number (V2 format).
    
    Args:
        filename: Path to the file to check
        
    Returns:
        str: 'v1', 'v2'
    """
    with open(filename, 'rb') as file_handle:
        first_four_bytes: bytes = file_handle.read(4)
        
    if len(first_four_bytes) < 4:
        return 'unknown'    
    # Unpack as little-endian unsigned integer
    value: int = struct.unpack('<L', first_four_bytes)[0]    
    # Check for specific magic numbers first
    if value == 0x48414148:  # "HAAH" - file header start
        return 'v2'    
    if (value & 0xFFFF) == 0xA55E:  # Sync header signature
        return 'v2'    
    # Check if it could be a reasonable Unix timestamp
    # Unix timestamps are typically between:
    # - 1970-01-01 (0) and some future date
    # - Reasonable range: 1970-01-01 to 2100-01-01
    min_timestamp: int = 0          # 1970-01-01 00:00:00 UTC
    max_timestamp: int = 4102444800 # 2100-01-01 00:00:00 UTC 
    if min_timestamp <= value <= max_timestamp:
        return 'v1'    
    return 'unknown'