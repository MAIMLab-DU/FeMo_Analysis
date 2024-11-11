import uuid
from urllib.parse import urlparse
from decimal import Decimal
from datetime import datetime


def generate_uuid_and_timestamp():
    # Generate a UUID
    generated_uuid = str(uuid.uuid4())

    # Get the current timestamp in ISO format
    current_timestamp = datetime.now().isoformat()

    return generated_uuid, current_timestamp


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

    # Extract the bucket name and the filename
    if parsed_url.scheme == 's3':
        path_parts = parsed_url.path.lstrip('/').split('/', 1)
        bucket_name = path_parts[0]
        filename = path_parts[1] if len(path_parts) > 1 else ''
        return bucket_name, filename
    else:
        raise ValueError(f"Invalid S3 path: {s3_path}")