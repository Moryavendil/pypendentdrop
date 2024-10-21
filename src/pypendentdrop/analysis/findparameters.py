from typing import Tuple, Union, Optional, Dict, Any, List
import math
import numpy as np
from scipy.integrate import odeint, trapezoid
from scipy.optimize import minimize
import time

from .. import error, warning, info, debug, trace


Fitparams = List[float]

# paremeters
def talk_params(fitparams:Fitparams, px_per_mm:float, talkfn=print, name=None) -> None:
    """
    Mainly a debug function, this displays the value of the parameters in a human readable form in the console.

    :param fitparams:
    :param px_per_mm:
    :return:
    """
    if name is None:
        name = 'Current'
    gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = fitparams
    talkfn(f'{name} parameters:')
    talkfn(f'\t\tpx_per_mm: {round(px_per_mm, 2) if px_per_mm is not None else None} px/mm')
    talkfn(f'\t\tgravity_angle: {round(gravity_angle * 180 / np.pi, 2) if gravity_angle is not None else None} deg')
    talkfn(f'\t\tx_tip_position: {round(x_tip_position, 2) if x_tip_position is not None else None} px')
    talkfn(f'\t\ty_tip_position: {round(y_tip_position, 2) if y_tip_position is not None else None} px')
    talkfn(f'\t\tr0: {round(r0_mm, 3) if r0_mm is not None else None} mm (= {round(r0_mm * px_per_mm, 1) if r0_mm is not None else None} px)')
    talkfn(f'\t\tcapillary_length: {round(capillary_length_mm, 3) if capillary_length_mm is not None else None} mm')

def image_centre(image:np.ndarray) -> np.ndarray:
    """
    Returns the (x, y) coordinates of the center of an image (around which it is pertinent to rotate the contour).

    :param image:
    :return:
    """
    return np.array(image.shape[1::-1]) / 2

def getrotationandscalematrix(centre, angle:float=0., scalefactor:float=1.):
    """
    Gets the matrix allowing the rotation and scaling of a contour (or an image) around a point.
    Shamelessly inspired from OpenCV's cv2.getRotationMatrix2D.

    :param centre:
    :param angle:
    :param scalefactor:
    :return:
    """
    c = math.cos(angle) * scalefactor
    s = math.sin(angle) * scalefactor
    return np.array([[c, s, (1-c)*centre[0] - s*centre[1]],
                     [-s, c, s*centre[0] + (1-c)*centre[1]]])

def rotate_and_scale(contour, angle:float=0., centre:Optional[Union[Tuple, List, np.ndarray]]=None, scalefactor=1.):
    """
    Rotates and/or scales a contour around a center **centre**.

    :param contour:
    :param angle:
    :param centre:
    :param scalefactor:
    :return:
    """
    if centre is None: # centre of rotation
        centre = (0, 0)
    rot_mat = getrotationandscalematrix(centre, angle=angle, scalefactor=scalefactor)
    # print(f'DBG: rot_mat {rot_mat.shape}:', rot_mat)
    # print(f'DBG: Contour shape: {np.array(contour).shape}')
    x_rotated = rot_mat[0,0] * contour[0] + rot_mat[0, 1] * contour[1] + rot_mat[0, 2]
    y_rotated = rot_mat[1,0] * contour[0] + rot_mat[1, 1] * contour[1] + rot_mat[1, 2]
    return x_rotated, y_rotated

def estimate_parameters(image_centre, contour:np.ndarray, px_per_mm) -> Fitparams:
    """
    Estimates the parameters crudely using (hopefully) robust techniques.

    :param image_centre:
    :param contour:
    :param px_per_mm:
    :return:
    """
    ### INITIAL ESTIMATION OF THE PARAMETERS

    ### ANGLE OF GRAVITY
    gravity_angle:float = 0.
    try:
        slope, intercept = np.polyfit(contour[1], contour[0], deg=1)
        abc0 = np.array([slope, -1., intercept])
        # abc0 = np.array([0., -1., contour[0].mean()])

        def dist(abc, contour:np.ndarray) -> float:
            return np.sum((abc[0]*contour[1] + abc[1]*contour[0] +abc[2])**2/(abc[0]**2+abc[1]**2))

        # print(f'DBG: Initial slope (numpy): {slope}, dist = {np.sqrt(dist(abc0, contour))}')

        bestline = minimize(dist, abc0, args=(contour))
        a, b, c = bestline.x

        # print(f'DBG: Slope (best line): {-a/b}, dist = {np.sqrt(dist(bestline.x, contour))}')

        gravity_angle = np.arctan(-a/b)
        # we do a trick to have the angle between -90 and 90 deg
        gravity_angle = (np.pi/2 + np.arctan2(-a, b))%(np.pi) - np.pi/2

        if np.abs(gravity_angle*180/np.pi) > 60:
            print(f'WARN: the angle of gravity was detected to {round(gravity_angle*180/np.pi, 2)} deg.')
            print(f'WARN: This is likely an error so I put it to 0.')
            gravity_angle = 0
    except:
        print("WARN: couldn't get gravity angle. Falling back to", gravity_angle)

    # print(f"gravity angle (initial estimate) = {initial_estimate_parameters['gravity_angle']*180/np.pi} deg")

    # Now we need to rotate the contour in order to correctly estimate the other parameters
    contour_tiltcorrected = rotate_and_scale(contour, angle=-gravity_angle, centre=image_centre)

    ### Position of the tip

    # we guess it from the tilt-corrected contour
    y_tip_position_tiltcorrected = contour_tiltcorrected[1].max()
    x_tip_position_tiltcorrected = np.mean(contour_tiltcorrected[0][np.argmax(contour_tiltcorrected[1])])

    # we translate that to the real contour
    x_tip_position, y_tip_position = rotate_and_scale([x_tip_position_tiltcorrected, y_tip_position_tiltcorrected],
                                                      angle=gravity_angle, centre=image_centre)


    # print(f"y tip position (initial estimate) = {initial_estimate_parameters['y_tip_position']} px")
    # print(f"x tip position (initial estimate) = {initial_estimate_parameters['x_tip_position']} px")

    ### Radius of curvature

    # coarse estimate using the tilt-corrected contour : max radius of the drop
    r0_px_coarse = (contour_tiltcorrected[0].max() - contour_tiltcorrected[0].min()) / 2
    try:
        # finer estimation : we take the points near the tip (at a distance < r0_estimation_radius, in px) and do a 2nd order polynomial fitparams_handpicked
        r0_estimation_radius = r0_px_coarse/4 # this is tipNeighbourhood is pendent drop
        nearthetip = (contour_tiltcorrected[0] - x_tip_position_tiltcorrected)**2 + (contour_tiltcorrected[1] - y_tip_position_tiltcorrected)**2 < r0_estimation_radius**2
        x_contour_neartip = contour_tiltcorrected[0][nearthetip]
        y_contour_neartip  = contour_tiltcorrected[1][nearthetip]
        curvature_px = np.polyfit(x_contour_neartip, y_contour_neartip, 2)[0]
        r0_px = np.abs(1/curvature_px)/2
    except:
        r0_px = r0_px_coarse
    r0_mm = r0_px / px_per_mm

    # print(f"r0 (initial estimate) = {initial_estimate_parameters['r0']} mm")

    capillary_length_mm:float = 2.7 # clueless guess : the fluid is pure water

    fitparams_init = [gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm]

    talk_params(fitparams_init, px_per_mm=px_per_mm, talkfn=trace, name='estimate_parameter: Result of estimation')

    return fitparams_init

### profile computing

def deriv(s_, rpkz):
    """
    The derivative along the arc of the variables.

    :param s_: (unused) The integration coordinate, the dimensionless arclength s
    :param rpkz: 4-array : [R, Psi, Kappa=dpsi/ds, Z]
    :return:
    """
    c = math.cos(rpkz[1])
    s = math.sin(rpkz[1])
    oneoverr = 1/rpkz[0]
    return [c, rpkz[2], -s + c * (s*oneoverr - rpkz[2])*oneoverr, s]

def greater_possible_zMax(tipRadius:float) -> float:
    """
    The max value zmax can take, from Daerr, pendent drop.

    :param tipRadius:
    :return:
    """
    return min(3.5, 3 / tipRadius) if tipRadius > .5 else 2 * tipRadius * (1 + 1.55*tipRadius)

def compute_nondimensional_profile(tipRadius:float, ds:float = 1e-3, approxLimit:float = 0.2, zMax:float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the nondimensional profile, starting at the bottom and up to an height zMax

    About the default value of ds:
    The length in px of the contour is L. We want around 2 points per pixel, and the length of the dimensionless drop will be
    s = L / (px_per_mm * capillarylength_in_mm).
    Thus taking a typical 100 px/mm (microscopic resolution is 1000) and a capillary length of 2.5, we have
    ds = s / (2 * L) = 1 / (2 * 100 * 2.5) = 2e-3
    To be safe, we take ds = 1e-3 by default.
    If using high resolution camera + excellent optics + optimal conditions, one might want it smaller.

    About the default value of approxLimit:
    It defines the fraction of capillary length or tip radius (whichever is smaller) up to which we use the approximate solution.
    Following pendent drop, we take it equal to 0.2

    About the solver : We use LSODA form LAPACK (scipy.integrate's odeint). Good ol' Fortran never dissapoints.

    :param tipRadius: The dimensionless tip radius r_0/l_c
    :param ds:
    :param approxLimit:
    :param zMax:
    :return:
    """
    if zMax is None:
        zMax = greater_possible_zMax(tipRadius)

    # print('Compute ND profile for tipRadius=', tipRadius)

    ### NEAR THE TIP (singularity handling by asymptotic development)

    # curvilinear coordinate where we switch from the approximate solution to integration
    sLimit = approxLimit * min(1., tipRadius) # sLimit << (Cap length, Tip radisu)

    # the curvilinear coordinates we will use
    s_neartip = np.arange(0, sLimit, ds, dtype=float)

    RPKZ_neartip = np.empty((len(s_neartip), 4))

    # use the approximation near the tip
    RPKZ_neartip[:, 0] = tipRadius * np.sin(s_neartip/tipRadius) + s_neartip**5/(40*tipRadius**2)
    RPKZ_neartip[:, 1] = s_neartip * (1 - s_neartip**2 / 8) / tipRadius
    RPKZ_neartip[:, 2] = (1 - 3 * s_neartip**2 / 8) / tipRadius
    RPKZ_neartip[:, 3] = s_neartip**2 * ( 1 + (1/16 - 1/(12*tipRadius**2)) * s_neartip**2 ) / (2*tipRadius)

    # AWAY FROM THE TIP
    N_awayfromtip = int(5 / ds)+1
    s_awayfromtip = np.arange(N_awayfromtip) * ds + s_neartip[-1]

    RPKZ_initial = RPKZ_neartip[-1, :]

    RPKZ_awayfromtip:np.ndarray = odeint(deriv, RPKZ_initial, s_awayfromtip, tfirst=True)
    # # if anyone wants to use RK45 explicitely (works but slower because the new and shiny solve_ivp is slower than the good ol' odeint):
    # RPKZ_awayfromtip = solve_ivp(deriv, [s_awayfromtip[0], s_awayfromtip[-1]], RPKZ_initial, method='RK45', t_eval=s_awayfromtip)
    # RPKZ_awayfromtip = RPKZ_awayfromtip.T

    Nmax_zMax_attained = N_awayfromtip if RPKZ_awayfromtip[-1, 3] <= zMax else np.argmax(RPKZ_awayfromtip[:, 3] > zMax)
    Nmax_axisymmetry = N_awayfromtip if RPKZ_awayfromtip[:, 0].min() > 0 else np.argmax(RPKZ_awayfromtip[:, 0] <= 0)
    Nmax_goingup = N_awayfromtip if RPKZ_awayfromtip[:, 1].min() > 0 else np.argmax(RPKZ_awayfromtip[:, 1] <= 0)
    Nmax_neck = np.argmin(np.maximum( RPKZ_awayfromtip[1:,1] > np.pi/2 , RPKZ_awayfromtip[:-1,1] <= np.pi/2))
    if Nmax_neck == 0: Nmax_neck = N_awayfromtip

    RPKZ_awayfromtip = RPKZ_awayfromtip[:min(Nmax_zMax_attained, Nmax_axisymmetry, Nmax_goingup, Nmax_neck),:]

    # combine everything
    RPKZ = np.concatenate((RPKZ_neartip, RPKZ_awayfromtip), axis=0)

    return RPKZ[:, 0], RPKZ[:, 3]

def integrated_contour(px_per_mm:float, fitparams:Fitparams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gives the computed profile in pixel coordinates.

    :param px_per_mm:
    :param fitparams:
    :return: The computed profile in pixel coordinates
    """
    gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = fitparams

    tipRadius = r0_mm / capillary_length_mm

    # hence the profile
    R, Z = compute_nondimensional_profile(tipRadius)

    # FOR DISPLAY : WE MAKE Rdim AND Zdim, TO SHOW ON IMAGE

    # we scale, then rotate (these two can be exchanged), then move
    Rdim, Zdim = R.copy(), Z.copy()

    # complete, using axisymmetry
    Rdim = np.concatenate((-Rdim[::-1], Rdim))
    Zdim = np.concatenate(( Zdim[::-1], Zdim))

    # Reverse Z because we have this weird image coordinated system
    Zdim *= -1
    # scaling (first by capillary length to dimensionalize, then by pizel size to fit the image)
    # rotate, around the tip at (0,0)
    Rdim, Zdim = rotate_and_scale([Rdim, Zdim], angle=gravity_angle, scalefactor=capillary_length_mm * px_per_mm)

    # moving
    Rdim = Rdim + x_tip_position
    Zdim = Zdim + y_tip_position

    return Rdim, Zdim

#compare computed profile to real profile

def compare_profiles(fitparams:Fitparams, contour, px_per_mm:float) -> float:
    gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = fitparams

    tipRadius = r0_mm / capillary_length_mm

    # hence the profile
    R, Z = compute_nondimensional_profile(tipRadius)

    # FOR COMPUTE THE DIFF : we take it backward
    XY = contour.copy()

    #moving
    XY[0] -= x_tip_position
    XY[1] -= y_tip_position

    #  rotating and scaling
    XY = rotate_and_scale(XY, angle=-gravity_angle, scalefactor=-1 / (capillary_length_mm * px_per_mm))

    # cutting off :
    XY = np.take(XY, np.where(XY[1] < Z.max())[0], axis=1)

    # separating the two sides
    rightside = XY[0] > 0
    X1, Y1 = np.take(XY, np.where(rightside)[0], axis=1)
    # X2, Y2 = -X[X < 0], Y[X < 0]
    # X2, Y2 = XY[:, np.bitwise_not(rightside)
    X2, Y2 = np.take(XY, np.where(np.bitwise_not(rightside))[0], axis=1)
    X2 *= -1

    # the differences
    R1 = np.interp(Y1, Z, R) # the radius corresponding to the side 1
    R2 = np.interp(Y2, Z, R) # the radius corresponding to the side 2

    R1[Y1 < Z.min()] *= 0
    R2[Y2 < Z.min()] *= 0
    DX1 = X1 - R1
    DX2 = X2 - R2

    chi2 = np.abs(trapezoid(DX1**2, Y1)) + np.abs(trapezoid(DX2**2, Y2))
    return chi2

# compute the optimal profile

def optimize_profile(contour:np.ndarray, px_per_mm:float, parameters_initialguess:Fitparams,
                     to_fit:Optional[List[bool]]=None,
                     maxiter:Optional[int]=None, method:Optional[str]=None) -> Tuple[bool, Fitparams]:
    """

    :param contour:
    :param px_per_mm:
    :param parameters_initialguess:
    :param to_fit: True if the parameter is to be fitted, False if it is to be fixed
    :param maxiter:
    :param method: 'Nelder-Mead' (faster) or 'Powell' (pendent-drop like)
    :return:
    """
    if method is None:
        method = 'Nelder-Mead'
    if to_fit is None:
        to_fit = [True, True, True, True, True]

    default_bounds = [(None, None),  # gravity_angle
                      (None, None),  # y_tip_position
                      (None, None),  # x_tip_position
                      (0, None),  # r0_mm
                      (0, None)] # capillary length (mm)

    bounds = [default_bounds[i] if to_fit[i] else (parameters_initialguess[i], parameters_initialguess[i]) for i in range(len(parameters_initialguess))]

    # we remove the top 5 pixels, i.e. the points too close to the top edge
    gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = parameters_initialguess
    nottooclosefromthetop = contour[1]-contour[1].min() > 5/np.cos(gravity_angle)
    contour_opti = contour[:, nottooclosefromthetop]

    options={}
    if maxiter is not None:
        options['maxiter'] = maxiter

    # pendent drop uses Powell's method with maxiter = 100 (typically maxiter = 10),
    # but I found the Nelder-Mead algorithm to be 25% faster ?
    # for method = Nelder-Mead, options={'adaptive':False, 'disp':False}

    t1 = time.time()
    minimization = minimize(compare_profiles, x0=np.array(parameters_initialguess), args=(contour_opti, px_per_mm),
                            bounds=bounds, method='Nelder-Mead', options=options)
    t2 = time.time()
    debug(f'Optimisation time: {int((t2-t1)*1000)} ms')

    optimization_success = minimization.success

    if not(minimization.success):
        info('Minimizaton failed')
        debug(f'Minimization unsuccessful: {minimization.message}')
        return False, parameters_initialguess

    debug(f'Minimization successful ({minimization.nit} iterations, {minimization.nfev} calls to function)')
    trace(f'Minimization message: {minimization.message}')

    return True, minimization.x