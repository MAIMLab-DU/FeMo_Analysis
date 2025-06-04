import numpy as np
from datetime import datetime, timedelta
from urllib.parse import urlparse


def extract_s3_details(s3_path: str):
    """Extracts the bucket name and filename from an S3 path."""
    # Parse the S3 path
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc
    filename = parsed_url.path.lstrip('/')

    # Extract the bucket name and the filename
    if bucket_name == '':
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    return bucket_name, filename


def extract_events(arr: np.ndarray, start_time: str,
                   time_per_sample: float, event_type: str) -> list[dict]:
    """Extracts events from a numpy array."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")
    if len(np.unique(arr)) > 2:
        raise ValueError("Input array must contain only binary values (0 or 1).")
    
    # Convert start_time string to datetime object
    if isinstance(start_time, str):
        # Handle ISO format with 'Z' suffix
        start_time_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    elif isinstance(start_time, datetime):
        start_time_dt = start_time
    else:
        raise TypeError("start_time must be a string in ISO format or datetime object")
    
    events = []
    diff = np.diff(np.concatenate(([0], arr, [0])))
    start_indices, end_indices = np.where(diff == 1)[0], np.where(diff == -1)[0]
    
    for start_idx, end_idx in zip(start_indices, end_indices):
        event_start_time = start_time_dt + timedelta(seconds=start_idx * time_per_sample)
        event_end_time = start_time_dt + timedelta(seconds=end_idx * time_per_sample)
        events.append({
            'event_type': event_type,
            'start_t': event_start_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            'end_t': event_end_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        })
    return events