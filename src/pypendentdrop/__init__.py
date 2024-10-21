from typing import Tuple, Union, Optional, Dict, Any, List

__version__ = '1.0.0.dev1'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import logging
logger = logging.getLogger(__name__)
def trace(msg:str):
    if hasattr(logging, 'TRACE'):
        logging.getLogger(__name__).trace(msg)

logger.addHandler(logging.NullHandler()) # so that messages are not shown in strerr

# # # REMOVE THIS AT ALL COSTS
# def addLoggingLevel(levelName, levelNum, methodName=None):
#     """
#     Comprehensively adds a new logging level to the `logging` module and the
#     currently configured logging class.
#
#     `levelName` becomes an attribute of the `logging` module with the value
#     `levelNum`. `methodName` becomes a convenience method for both `logging`
#     itself and the class returned by `logging.getLoggerClass()` (usually just
#     `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
#     used.
#
#     To avoid accidental clobberings of existing attributes, this method will
#     raise an `AttributeError` if the level name is already an attribute of the
#     `logging` module or if the method name is already present
#     """
#     if not methodName:
#         methodName = levelName.lower()
#
#     if hasattr(logging, levelName):
#         raise AttributeError('{} already defined in logging module'.format(levelName))
#     if hasattr(logging, methodName):
#         raise AttributeError('{} already defined in logging module'.format(methodName))
#     if hasattr(logging.getLoggerClass(), methodName):
#         raise AttributeError('{} already defined in logger class'.format(methodName))
#
#     # This method was inspired by the answers to Stack Overflow post
#     # http://stackoverflow.com/q/2183233/2988730, especially
#     # http://stackoverflow.com/a/13638084/2988730
#     def logForLevel(self, message, *args, **kwargs):
#         if self.isEnabledFor(levelNum):
#             self._log(levelNum, message, args, **kwargs)
#     def logToRoot(message, *args, **kwargs):
#         logging.log(levelNum, message, *args, **kwargs)
#
#     logging.addLevelName(levelNum, levelName)
#     setattr(logging, levelName, levelNum)
#     setattr(logging.getLoggerClass(), methodName, logForLevel)
#     setattr(logging, methodName, logToRoot)
# addLoggingLevel('TRACE', logging.DEBUG - 5)
# logging.basicConfig(level=logging.TRACE,
#                     format=f"%(asctime)s {bcolors.OKGREEN}[%(levelname)s]{bcolors.ENDC} %(message)s",
#                     handlers=[
#                         logging.FileHandler(filename='ppdlog.log', mode='w'),
#                         logging.StreamHandler()
#                     ]
#                     ) # REMOVE THIS

logger.debug('pypendentdrop loaded')


trace('We have trace, yay')

###### ANALYZE
from .analysis.fetchimage import *
from .analysis.getcontour import *
from .analysis.findparameters import *

