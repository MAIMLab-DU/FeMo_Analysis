import os
from ._utils import get_logger

LOG_PATH = 'logs/'

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

_logFile = os.path.join(LOG_PATH, 'app.log')

logger = get_logger(__name__, _logFile)