import os
import warnings
import logging

# Set environment variables BEFORE any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['AUTOGRAPH_VERBOSITY'] = '0'  # Disable AutoGraph warnings

# Comprehensive warning suppression
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress TensorFlow and Keras logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow.python.util.deprecation').setLevel(logging.ERROR)


__version__ = "2.0.0"

__all__ = (
    "__version__"
)
