import hashlib
import numpy as np


def gen_hash(input_string: str) -> str:
    # Use SHA-256 to generate a hash of the input string
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))  # Convert the input string to bytes and hash it
    full_hash = sha256.hexdigest()[-4:]  # Get the full hash as a hexadecimal string
    full_hash = int(''.join([str(ord(c)) for c in full_hash]))
    return full_hash


def normalize_features(data: np.ndarray,
                       mu: np.ndarray|None = None,
                       dev: np.ndarray|None = None):
    if mu is None:
        mu = np.mean(data, axis=0)
    norm_feats = data - mu
    if dev is None:
        dev = np.max(norm_feats, axis=0) - np.min(norm_feats, axis=0)
    # Avoid division by zero by adding a small epsilon
    norm_feats = norm_feats / dev

    return norm_feats[:, ~np.any(np.isnan(norm_feats), axis=0)], mu, dev