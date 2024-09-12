import logging
from logging.handlers import RotatingFileHandler

_log_format = "%(asctime)s - [%(levelname)s] - [%(threadName)s - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d)] - %(message)s"

def get_file_handler(_logFile):
    # file_handler = logging.FileHandler("app.log", mode='w')
    file_handler = RotatingFileHandler(_logFile, mode='w', maxBytes=20*1024*1024,
                                       backupCount=15, encoding=None, delay=False)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler

def get_logger(name, _logFile):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_file_handler(_logFile))
    logger.addHandler(get_stream_handler())
    return logger