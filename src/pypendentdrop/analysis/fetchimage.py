from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np
from PIL import Image

from .. import error, warning, info, debug, trace


# import image and threshold
def import_image(file_path:Optional[str] = None) -> Tuple[bool, np.ndarray]:
    """Imports an image in grayscale.

    Tries to import the image at `file_path`.
    Returns True and the image in grayscale if the import succeeded.
    Returns False and a random 128 x 128 matrix with values 0-255 if the import failed.

    Parameters
    ----------
    file_path : str
        The path to the file to open. Can be None.

    Returns
    -------
    success : bool
        Whether of not the image could be imported.

    data : ndarray
        The image in grayscale.

    """
    success = False
    data = None
    if file_path is None:
        debug('import_image: File path provided is None. Failing to import')
    else:
        try:
            imagedata = Image.open(file_path)
            data = np.asarray(imagedata, dtype="float")
            if len(data.shape) > 2: # go to gray
                data = np.mean(data, axis=2)
            success = True
        except:
            # todo: add precise reason why it did not work
            warning(f'pypendentdrop.import_image: Could not import the image at path "{file_path}"')
    if not success:
        data = np.random.randint(0, 255, (128, 128))
    trace(f'import_image: Imported image {file_path} with success: {success}')
    return success, data
