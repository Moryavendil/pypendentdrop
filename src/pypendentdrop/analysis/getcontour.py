from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np
import warnings
from contourpy import contour_generator, LineType

from .. import error, warning, info, debug, trace

# Region OF Interest management
Roi = Optional[List[Optional[int]]]

# warnings.filterwarnings("ignore", category=RuntimeWarning)
def format_roi(data:np.ndarray, roi:Roi=None):
    if roi is None:
        roi = [None, None, None, None] # TLx, TLy, BRx, BRy
    height, width = data.shape

    tlx, tly, brx, bry = roi
    if tlx is None:
        trace('format_roi: TLX not provided.')
        tlx = 0
    else:
        if not(0 <= tlx < width):
            warning(f'TLX="{tlx}" does not verify 0 <= TLX < width={width}. Its was overriden: TLX=0')
            tlx = 0

    if tly is None:
        trace('format_roi: TLX not provided.')
        tly = 0
    else:
        if not(0 <= tly < height):
            warning(f'TLY="{tly}" does not verify 0 <= TLY < height={height}. Its was overriden: TLY=0')
            tly = 0

    if brx is None:
        trace('format_roi: BRX not provided.')
        brx = None
    else:
        if not(tlx <= brx < width):
            warning(f'BRX="{brx}" does not verify TLX={tlx} <= BRX < width={width}. Its was overriden: BRX=None (=width)')
            brx = None

    if bry is None:
        trace('format_roi: BRY not provided.')
        bry = None
    else:
        if not(tly <= bry < height):
            warning(f'BRY="{bry}" does not verify TLY={tly} <= BRY < height={height}. Its was overriden: BRX=None (=height)')
            brx = None

    trace(f'format_roi: {roi} -> {[tlx, tly, brx, bry]}')
    return [tlx, tly, brx, bry]
def otsu_intraclass_variance(image, threshold):
    """
    Otsu's intra-class variance.
    If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.
    """
    try:
        return np.nansum(
            [
                np.mean(cls) * np.var(image, where=cls)
                #   weight   ·  intra-class variance
                for cls in [image >= threshold, image < threshold]
            ]
        )
    except:
        return 0
    # NaNs only arise if the class is empty, in which case the contribution should be zero, which `nansum` accomplishes.

def otsu_threshold(data:np.ndarray) -> int:
    test_tresholds = np.arange(255, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        otsu_variance = np.array([otsu_intraclass_variance(data, test_treshold) for test_treshold in test_tresholds])

    best_threshold_otsu = int(test_tresholds[np.argmin(otsu_variance)])

    return best_threshold_otsu

def best_threshold(image:np.ndarray, roi:Roi=None) -> int:
    """
    Trying to find Otsu's most appropriate threshold for the image, falling back to 127 it it fails.

    :param image:
    :return:
    """
    roi = format_roi(image, roi=roi)
    try:
        threshold:int = otsu_threshold(image[roi[0]:roi[2], roi[1]:roi[3]])
    except:
        threshold = 127
        error('Encountered an error while computing the best threshold')
    trace(f'best_threshold: Best threshold for the selected region of the image is {threshold}')
    return threshold

def detect_contourlines(data:np.ndarray, level:Union[int, float], roi:Roi=None) -> List[np.ndarray]:
    """
    Gets a collection of lines that each a contour of the level **level** of the data.
    Each line is in line form, i.e. shape=(N,2)

    :param data:
    :param level:
    :param roi:
    :return:
    """
    trace('detect_contourlines: called')
    roi = format_roi(data, roi=roi)

    cont_gen = contour_generator(z=data[roi[1]:roi[3], roi[0]:roi[2]], line_type=LineType.Separate) # quad_as_tri=True

    lines = cont_gen.lines(level)

    for i_line, line in enumerate(lines):
        lines[i_line] = np.array(line) + np.expand_dims(np.array(roi[:2]), 0)

    return lines

def detect_main_contour(data:np.ndarray, level:Union[int, float], roi:Roi=None) -> np.ndarray:
    """
    Returns the longest lines from detect_contourlines in a contour-form: shape=(2, N)

    :param data:
    :param level:
    :param roi:
    :return:
    """
    lines = detect_contourlines(data, level, roi=roi)

    return np.array(lines[np.argmax([len(line) for line in lines])]).T
