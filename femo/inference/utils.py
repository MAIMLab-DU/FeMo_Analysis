import numpy as np
from typing import Optional
from scipy import ndimage


def smooth_signal(
        data: np.ndarray, cutoff: float = 3.0, fs: int = 1024, order: int = 2
    ) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter to smooth the input signal.

        Args:
            data (np.ndarray): Input signal array.
            cutoff (float): Cutoff frequency for the filter in Hz.
            fs (int): Sampling frequency in Hz.
            order (int): Order of the Butterworth filter.

        Returns:
            np.ndarray: Smoothed signal.
        """
        from scipy.signal import butter, filtfilt

        if data.size == 0 or data.size <= 3 * order:
            return np.array([], dtype=float)
        b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        return filtfilt(b, a, data)


def smooth_labels(
    labels: np.ndarray, 
    window_time: int = 1, 
    sensor_freq: int = 1024
) -> np.ndarray:
    """
    Ultra-fast label smoothing using ndimage for categorical labels. 
    This function uses a uniform filter to count occurrences of each label within a moving window,
    then assigns the most frequent label in that window.
    Note that to reliably remove spikes of length `n` seconds, the window time should be at least `2*(n+1)+n` seconds.
    
    Args:
        labels: Array of categorical orientation labels to smooth (0, 1, 2, 3, 5)
        window_time: Window size in seconds for smoothing
        sensor_freq: Sampling frequency in Hz
        
    Returns:
        Smoothed orientation labels array with same categorical values
    """
    window_size_samples: int = int(window_time * sensor_freq)
    if window_size_samples % 2 == 0:
        window_size_samples += 1
    
    array_length: int = len(labels)
    
    # Handle edge cases
    if window_size_samples >= array_length:
        unique_values, counts = np.unique(labels, return_counts=True)
        global_mode: int = unique_values[np.argmax(counts)]
        return np.full_like(labels, global_mode)
    
    # Get unique categorical labels
    unique_labels: np.ndarray = np.unique(labels)
    num_unique_labels: int = len(unique_labels)
    
    if num_unique_labels <= 1:
        return labels.copy()
    
    # Create one-hot encoding for each unique label
    label_counts: np.ndarray = np.zeros((num_unique_labels, array_length), dtype=np.float32)
    
    for label_idx, label_value in enumerate(unique_labels):
        # Create binary mask for this label
        label_mask: np.ndarray = (labels == label_value).astype(np.float32)
        
        # Apply uniform filter to get moving counts
        label_counts[label_idx] = ndimage.uniform_filter1d(
            label_mask, 
            size=window_size_samples, 
            mode='nearest'
        )
    
    # Find label with highest count at each position
    max_count_indices: np.ndarray = np.argmax(label_counts, axis=0)
    smoothed_labels: np.ndarray = unique_labels[max_count_indices]
    
    return smoothed_labels.astype(labels.dtype)


def classify_posture(
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> int:
    """
    Classify posture based on roll and pitch angles or x, y, z coordinates.

    Args:
        roll (Optional[float]): Roll angle in degrees.
        pitch (Optional[float]): Pitch angle in degrees.
        x (Optional[float]): X-axis value.
        y (Optional[float]): Y-axis value.
        z (Optional[float]): Z-axis value.

    Returns:
        int: Posture class label.
            0 - upright
            1 - lying flat
            2 - lying right
            3 - lying left
            5 - unknown
    """
    if roll is not None and pitch is not None:
        threshold = 45
        if (90-threshold) <= abs(pitch) <= (90+threshold):
            return 0
        if abs(roll) <= threshold:
            return 1
        if roll > 0:
            return 2
        if roll < 0:
            return 3
        return 5

    if all(coord is not None for coord in (x, y, z)):
        abs_values: list[float] = [abs(x), abs(y), abs(z)]
        sorted_indices: np.ndarray = np.argsort(abs_values) # ascending order
        max_idx: int = sorted_indices[-1]
        
        # Check component with highest magnitude
        if max_idx == 0:  # x component dominates
            if x < 0:
                return 0
            elif abs(abs(y)-abs(z)) < 0.1:
                return 1
            else:
                return 3 if y < 0 else 2
        if max_idx == 1:  # y component dominates
            return 2 if y < 0 else 3
        if max_idx == 2:  # z component dominates
            return 1
    return 5


