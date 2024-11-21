import os
from ._utils import get_logger

LOG_PATH = 'logs/'

os.makedirs(LOG_PATH, exist_ok=True)

_logFile = os.path.join(LOG_PATH, 'app.log')

LOGGER = get_logger(__name__, _logFile)