__version__ = "0.0.1.dev11"

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # so that messages are not shown in strerr
def error(msg:str):
    logger.error(msg)
def warning(msg:str):
    logger.warning(msg)
def info(msg:str):
    logger.info(msg)
def debug(msg:str):
    logger.debug(msg)
def trace(msg:str):
    try:
        logger.trace(msg)
    except:
        pass

debug(f'pypendentdrop version {__version__} loaded')

###### ANALYZE
from .analysis.fetchimage import *
from .analysis.getcontour import *
from .analysis.findparameters import *

