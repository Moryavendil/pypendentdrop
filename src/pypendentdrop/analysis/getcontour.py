from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np
from contourpy import contour_generator, LineType


import logging
logger=logging.getLogger(__name__)
def trace(msg:str):
    if hasattr(logging, 'TRACE'):
        logging.getLogger(__name__).trace(msg)

# Region OF Interest management
Roi = Optional[List[Optional[int]]]

def format_roi(data:np.ndarray, roi:Roi=None):
    if roi is None:
        roi = [None, None, None, None] # TLx, TLy, BRx, BRy
    height, width = data.shape
    trace(f'format_roi: Formatting roi={roi}=[TLX, TLY, BRX, BRY]')

    tlx, tly, brx, bry = roi
    if tlx is None:
        trace('format_roi: TLX not provided.')
        tlx = 0
    else:
        if not(0 <= tlx < width):
            logger.warning(f'TLX="{tlx}" does not verify 0 <= TLX < width. Its was overriden: TLX=0')
            tlx = 0

    if tly is None:
        trace('format_roi: TLX not provided.')
        tly = 0
    else:
        if not(0 <= tly < height):
            logger.warning(f'TLY="{tly}" does not verify 0 <= TLY < height. Its was overriden: TLY=0')
            tly = 0

    if brx is None:
        trace('format_roi: BRX not provided.')
        brx = None
    else:
        if not(tlx <= brx < width):
            logger.warning(f'BRX="{brx}" does not verify TLX <= BRX < width. Its was overriden: BRX=None (=width)')
            brx = None

    if bry is None:
        trace('format_roi: BRY not provided.')
        bry = None
    else:
        if not(tly <= bry < height):
            logger.warning(f'BRY="{bry}" does not verify TLY <= BRY < height. Its was overriden: BRX=None (=height)')
            brx = None

    trace(f'format_roi: Formatted ROI={[tlx, tly, brx, bry]}')
    return [tlx, tly, brx, bry]

def best_threshold(data:np.ndarray, roi:Roi=None) -> int:
    """
    TO BE IMPLEMENTED

    Trying to find Otsu's most appropriate threshold for the image, falling back to 127 it it fails.

    :param data:
    :return:
    """
    roi = format_roi(data, roi=roi)
    logger.warning('best_threshold: NOT IMPLEMENTED')
    return 127

def find_contourLines(data:np.ndarray, level:Union[int, float], roi:Roi=None) -> List[np.ndarray]:
    """
    Gets a collection of lines that each a contour of the level **level** of the data.
    Each line is in line form, i.e. shape=(N,2)

    :param data:
    :param level:
    :param roi:
    :return:
    """
    roi = format_roi(data, roi=roi)
    # print('DEBUG: Generating contour lines')

    cont_gen = contour_generator(z=data[roi[1]:roi[3], roi[0]:roi[2]], line_type=LineType.Separate) # quad_as_tri=True

    lines = cont_gen.lines(level)

    for i_line, line in enumerate(lines):
        lines[i_line] = np.array(line) + np.expand_dims(np.array(roi[:2]), 0)

    return lines

def find_mainContour(data:np.ndarray, level:Union[int, float], roi:Roi=None) -> np.ndarray:
    """
    Returns the longest lines from find_contourLines in a contour-form: shape=(2, N)

    :param data:
    :param level:
    :param roi:
    :return:
    """
    lines = find_contourLines(data, level, roi=roi)

    return np.array(lines[np.argmax([len(line) for line in lines])]).T
