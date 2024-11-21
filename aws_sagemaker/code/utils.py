from urllib.parse import urlparse
from decimal import Decimal


def convert_floats_to_decimal(data):
    if isinstance(data, dict):
        return {k: convert_floats_to_decimal(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_floats_to_decimal(i) for i in data]
    elif isinstance(data, float):
        return Decimal(str(data))
    return data


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