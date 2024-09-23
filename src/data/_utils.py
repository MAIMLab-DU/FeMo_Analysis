import hashlib

def gen_hash(input_string: str) -> str:
    # Use SHA-256 to generate a hash of the input string
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))  # Convert the input string to bytes and hash it
    full_hash = sha256.hexdigest()[-4:]  # Get the full hash as a hexadecimal string
    full_hash = int(''.join([str(ord(c)) for c in full_hash]))
    return full_hash
