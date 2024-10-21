from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np
from PIL import Image

import logging
logger=logging.getLogger(__name__)
def trace(msg:str):
    if hasattr(logging, 'TRACE'):
        logging.getLogger(__name__).trace(msg)

# import image and threshold
def import_image(filePath:Optional[str] = None) -> Tuple[bool, np.ndarray]:
    """
    Imports an image from a path.
    Returns True and the image in gray scale if the image can be imported.
    Returns False and a random matrix if the import failed.

    :param filePath:
    :return:
    """
    success = False
    data = None
    trace(f'import_image: Trying to open {filePath}')
    if filePath is None:
        logger.debug('import_image: File path provided is None. Failing to import')
    else:
        try:
            imagedata = Image.open(filePath)
            data = np.asarray(imagedata, dtype="float")
            if len(data.shape) > 2: # go to gray
                data = np.mean(data, axis=2)
            success = True
        except:
            logger.warning(f'import_image: Could not import the image at {filePath}')
    if not success:
        data = np.random.randint(0, 255, (128, 128))
    return success, data
