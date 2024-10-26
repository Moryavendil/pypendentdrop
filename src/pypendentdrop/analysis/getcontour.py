from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np
import warnings
from contourpy import contour_generator, LineType

from .. import error, warning, info, debug, trace

# Region OF Interest management
Roi = Optional[List[Optional[int]]]

def format_roi(image:np.ndarray, roi:Roi=None) -> Roi:
    """Formats a ROI so that it is inside the image.

    A ROI is a 4-list (or 4-tuple) where the 2 first entries are the (x, y) coordinates of the top-left corner,
    and the 2 last entries are the (x, y) coordinates of the bottom right corner.
    This function checks that the corners are well-ordered and inside the image.
    None is treated as min(=0) for the TL corner or max(=width or height) for the BR corner.

    Parameters
    ----------
    image : ndarray

    roi : Roi
        Initial ROI. Can be None, or [None, None, None, None]

    Returns
    -------
    roi : Roi
        The formatted ROI

    """
    if roi is None:
        roi = [None, None, None, None] # TLx, TLy, BRx, BRy
    height, width = image.shape

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
        if not(tlx < brx <= width):
            warning(f'BRX="{brx}" does not verify TLX={tlx} < BRX <= width={width}. Its was overriden: BRX=None (=width)')
            brx = None

    if bry is None:
        trace('format_roi: BRY not provided.')
        bry = None
    else:
        if not(tly < bry <= height):
            warning(f'BRY="{bry}" does not verify TLY={tly} < BRY <= height={height}. Its was overriden: BRX=None (=height)')
            brx = None

    trace(f'format_roi: {roi} -> {[tlx, tly, brx, bry]}')
    return [tlx, tly, brx, bry]

def otsu_intraclass_variance(image:np.ndarray, threshold:Union[int, float]) -> float:
    """Otsu's intra-class variance.

    If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.

    Parameters
    ----------
    image : ndarray
        The image

    threshold : float

    Returns
    -------
    variance : float

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
        return 0.
    # NaNs only arise if the class is empty, in which case the contribution should be zero, which `nansum` accomplishes.

def otsu_threshold(image:np.ndarray) -> int:
    """Otsu's optimal threshold for an image.

    Computes `Otsu's intraclass variance <pypendentdrop.otsu_intraclass_variance>` for all integers 0-225 and returns the best threshold.

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    thershold : int

    """
    test_tresholds = np.arange(255, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        otsu_variance = np.array([otsu_intraclass_variance(image, test_treshold) for test_treshold in test_tresholds])

    best_threshold_otsu = int(test_tresholds[np.argmin(otsu_variance)])

    return best_threshold_otsu

def best_threshold(image:np.ndarray, roi:Roi=None) -> float:
    """Finds the most appropriate threshold for the image.

    Trying to find Otsu's most appropriate threshold for the image, falling back to 127 it it fails.

    Parameters
    ----------
    image : ndarray
    roi : Roi, optional

    Returns
    -------
    threshold : float

    """
    roi = format_roi(image, roi=roi)
    try:
        threshold:int = otsu_threshold(image[roi[0]:roi[2], roi[1]:roi[3]])
    except:
        threshold = 127
        error('Encountered an error while computing the best threshold')
    trace(f'best_threshold: Best threshold for the selected region of the image is {threshold}')
    return threshold

def detect_contourlines(image:np.ndarray, level:float, roi:Roi=None) -> List[np.ndarray]:
    """Returns all the closed lines enclosing regions in ``image`` that are above ``level``.

    Returns a collection of lines that each a contour of the level ``level`` of the image.
    Each line is in line-form, i.e. shape=(N,2).

    Parameters
    ----------
    image : ndarray
    level : float
    roi : Roi, optional

    Returns
    -------
    lines : array_like
        A collection of ndarrays of shape (N, 2).

    """
    trace('detect_contourlines: called')
    roi = format_roi(image, roi=roi)

    cont_gen = contour_generator(z=image[roi[1]:roi[3], roi[0]:roi[2]], line_type=LineType.Separate) # quad_as_tri=True

    lines = cont_gen.lines(level)

    for i_line, line in enumerate(lines):
        lines[i_line] = np.array(line) + np.expand_dims(np.array(roi[:2]), 0)

    return lines

def detect_main_contour(image:np.ndarray, level:float, roi:Roi=None) -> np.ndarray:
    """Returns the main (longest) closed line enclosing a region in ``image`` that is above ``level``.

    Finds the longest of all `contour lines <pypendentdrop.detect_contourlines>` above a specific level,
    and returns its transposition, so that it is of shape (2, N).

    Parameters
    ----------
    image : ndarray
    level : float
    roi : Roi, optional

    Returns
    -------
    lines : ndarray
        An ndarray of shape (2, N).

    """
    lines = detect_contourlines(image, level, roi=roi)

    return np.array(lines[np.argmax([len(line) for line in lines])]).T
