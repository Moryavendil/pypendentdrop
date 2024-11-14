from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np
import warnings
from contourpy import contour_generator, LineType

from .. import error, warning, info, debug, trace

# Region OF Interest management
Roi = Optional[List[Optional[int]]]


def format_roi(image: np.ndarray, roi: Roi = None) -> Roi:
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
        roi = [None, None, None, None]  # TLx, TLy, BRx, BRy
    height, width = image.shape

    tlx, tly, brx, bry = roi
    if tlx is None:
        trace('format_roi: TLX not provided.')
        tlx = 0
    else:
        if not (0 <= tlx < width):
            warning(f'TLX="{tlx}" does not verify 0 <= TLX < width={width}. Its was overriden: TLX=0')
            tlx = 0

    if tly is None:
        trace('format_roi: TLX not provided.')
        tly = 0
    else:
        if not (0 <= tly < height):
            warning(f'TLY="{tly}" does not verify 0 <= TLY < height={height}. Its was overriden: TLY=0')
            tly = 0

    if brx is None:
        trace('format_roi: BRX not provided.')
        brx = None
    else:
        if not (tlx < brx <= width):
            warning(
                f'BRX="{brx}" does not verify TLX={tlx} < BRX <= width={width}. Its was overriden: BRX=None (=width)')
            brx = None

    if bry is None:
        trace('format_roi: BRY not provided.')
        bry = None
    else:
        if not (tly < bry <= height):
            warning(
                f'BRY="{bry}" does not verify TLY={tly} < BRY <= height={height}. Its was overriden: BRX=None (=height)')
            brx = None

    trace(f'format_roi: {roi} -> {[tlx, tly, brx, bry]}')
    return [tlx, tly, brx, bry]


def otsu_intraclass_variance(image:np.ndarray, threshold: Union[int, float]) -> float:
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

#TODO: Remove this
def otsu_threshold_slower(image:np.ndarray) -> int:
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

#TODO: Remove this
def otsu_threshold_faster(image:np.ndarray) -> int:
    # adapted from https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python
    # Because we did not want to import scikit.image of opencv
    pixel_number = image.shape[0] * image.shape[1]
    mean_weight = 1.0 / pixel_number
    his, bins = np.histogram(image, np.arange(0, 256 + 1))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:  # this is inefficient but only computed 254 times so should be ok
            final_thresh = t
            final_value = value

    return final_thresh


def threshold_otsu(image:np.ndarray) -> float:
    """Find the threshold value for a bimodal histogram using the Otsu method.

    If you have a distribution that is bimodal (AKA with two peaks, with a valley
    between them), then you can use this to find the location of that valley, that
    splits the distribution into two.

    From the SciKit Image threshold_otsu implementation:
    https://github.com/scikit-image/scikit-image/blob/70fa904eee9ef370c824427798302551df57afa1/skimage/filters/thresholding.py#L312
    """
    if image.min() == image.max():
        return float(image[0,0])
    counts, bin_edges = np.histogram(image.ravel(), bins=np.arange(image.min()-0.5, image.max()+.5+1, 1))
    bin_centers:np.ndarray = (bin_edges[1:] + bin_edges[:-1]) / 2
    trace(f'bin centers: {bin_centers}')

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    threshold:float = float(bin_centers[np.argmax(variance12)])
    return threshold

def auto_threshold(image: np.ndarray, roi: Roi = None) -> float:
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
    trace(f"auto_threshold: Trying to find Otsu's optimal threshold")
    try:
        # threshold:int = otsu_threshold_slower(image[roi[0]:roi[2], roi[1]:roi[3]])
        # threshold: int = otsu_threshold_faster(image[roi[0]:roi[2], roi[1]:roi[3]])
        threshold: int = int(np.rint(threshold_otsu(image[roi[0]:roi[2], roi[1]:roi[3]])))
    except:
        threshold = 127
        error('Encountered an error while computing the best threshold.')
    trace(f'auto_threshold: Best threshold for the selected region of the image is {threshold}')
    return threshold


def detect_contourlines(image: np.ndarray, level: float, roi: Roi = None) -> List[np.ndarray]:
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

    cont_gen = contour_generator(z=image[roi[1]:roi[3], roi[0]:roi[2]], line_type=LineType.Separate)  # quad_as_tri=True

    lines = cont_gen.lines(level)

    for i_line, line in enumerate(lines):
        lines[i_line] = np.array(line) + np.expand_dims(np.array(roi[:2]), 0)

    return lines


def detect_main_contour(image: np.ndarray, level: float, roi: Roi = None) -> np.ndarray:
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
