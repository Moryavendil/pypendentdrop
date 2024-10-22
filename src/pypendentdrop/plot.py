
from . import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.integrate import trapezoid

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.max_open_warning"] = 50

# plt.rcParams['pgf.texsystem'] = 'pdflatex' # use this if you have LaTeX
plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'figure.titlesize' : 10,
                     'axes.labelsize': 10,'axes.titlesize': 12,
                     'legend.fontsize': 10})

def plot_image_contour(ax, image:np.ndarray, contour:np.ndarray, parameters:Parameters, comment='', roi=None):
    roi = format_roi(image, roi)
    roi[2] = roi[2] or image.shape[1]
    roi[3] = roi[3] or image.shape[0]
    ax.set_title(f'Drop image and contour ({comment})')
    ax.imshow(image, cmap='gray')
    ax.plot([roi[0], roi[0], roi[2], roi[2], roi[0]], [roi[1], roi[3], roi[3], roi[1], roi[1]], lw=2, c='y', ls=':', label='ROI')

    xcontour, ycontour = contour[0], contour[1]
    ax.plot(xcontour, ycontour, c='lime', lw=2, label='Detected contour')

    gravity_angle = parameters.get_a_rad()
    x_tip_position, y_tip_position = parameters.get_xy_px()

    l = max(image.shape)
    ax.plot((x_tip_position + l * np.sin(-gravity_angle), x_tip_position - l * np.sin(-gravity_angle)), (y_tip_position - l * np.cos(-gravity_angle), y_tip_position + l * np.cos(-gravity_angle)),
            color='b', lw=2, ls='--', label=f'Direction of gravity')

    ax.scatter(x_tip_position, y_tip_position, s=50, fc='k', ec='lime', linewidths=2, label=f'Tip position', zorder=4)

    r0_px = parameters.get_r_px()

    drop_center_x = x_tip_position + r0_px * np.sin(-gravity_angle)
    drop_center_y = y_tip_position - r0_px * np.cos(-gravity_angle)
    e1 = patches.Arc((drop_center_x, drop_center_y), 2 * r0_px, 2 * r0_px,  # WARNING CONVENTION
                     theta1 = 0 - gravity_angle*180/np.pi, theta2 = 180 - gravity_angle*180/np.pi,
                     linewidth=2, fill=False, zorder=2, color='darkred', ls='--', label=f'Curvature')
    ax.add_patch(e1)

    Rd, Zd = integrated_contour(parameters)

    ax.plot(Rd, Zd, c='r', lw=2, label=f'Computed profile')

    ax.legend()
    ax.set_xlabel('x [px]')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylabel('y [px]')
    ax.set_ylim(image.shape[0], 0)

def plot_difference(axtop, axbot, contour, parameters:Parameters, comment=''):
    # axtop.set_title(f'chi2: {analyze.compute_gap_dimensionless(fitparams, contour, px_per_mm=px_per_mm)}')
    axtop.set_title(f'Comparison of detected contour and computed profile')

    fitparams:Fitparams = parameters.get_fitparams()

    gravity_angle, x_tip_position, y_tip_position, r0_px, capillary_length_px = fitparams

    tipRadius = r0_px / capillary_length_px

    # hence the profile
    R, Z = compute_nondimensional_profile(tipRadius)

    # FOR COMPUTE THE DIFF : we take it backward
    XY = contour.copy()

    #moving
    XY[0] -= x_tip_position
    XY[1] -= y_tip_position

    #  rotating and scaling
    XY = rotate_and_scale(XY, angle=-gravity_angle, scalefactor=-1 / capillary_length_px)

    # # cutting off :
    # XY = np.take(XY, np.where(XY[1] < Z.max())[0], axis=1)


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

    R1[Y1 > Z.max()] = R[np.argmax(Z)]
    R2[Y2 > Z.max()] = R[np.argmax(Z)]

    R1[Y1 < Z.min()] *= 0
    R2[Y2 < Z.min()] *= 0
    DX1 = X1 - R1
    DX2 = X2 - R2

    difference = np.abs(trapezoid(DX1**2, Y1)) + np.abs(trapezoid(DX2**2, Y2))
    # print(f'DGB: CHI2: {chi2}')

    ### AX ON TOP

    axtop.plot(Z, R, c='m', ls=':', lw=1)
    axtop.plot(Y1, R1, c='r', lw=1)
    axtop.plot(Y2, R2, c='r', lw=1, label=f'Computed contour ({comment})')
    axtop.plot(Y1, X1, c='lime', lw=1, ls='--', label='Drop contour (right side)')
    axtop.plot(Y2, X2, c='lime', lw=1, ls=':', label='Drop contour (left side)')

    axtop.legend()
    axtop.yaxis.tick_right()
    axtop.yaxis.set_label_position('right')
    axtop.set_xlim(min(Y1.min(), Y2.min(), 0), max(Y1.max(), Y2.max()))
    axtop.set_ylim(0, max(X1.max(), X2.max(), R.max()) * 1.2)
    axtop.set_xlabel('Z [dimensionless]')
    axtop.set_ylabel('R [dimensionless]')

    ### AX ON BOTTOM

    axbot.axhline(0, c='gray', alpha=.3)
    axbot.plot(Y1, DX1, ls='--', c='gray', label='Right side')
    axbot.plot(Y2, DX2, ls=':', c='gray', label='Left side')

    axbot.legend()
    axbot.yaxis.tick_right()
    axbot.yaxis.set_label_position('right')
    axbot.set_xlim(min(Y1.min(), Y2.min(), 0), max(Y1.max(), Y2.max()))
    bnd = max(np.abs(DX1).max(), np.abs(DX2).max()) * 1.2
    axbot.set_ylim(-bnd, bnd)
    axbot.set_xlabel('Z [dimensionless]')
    axbot.set_ylabel('Detected contour - computed profile')
    # axbot.spines['right'].set_color('gray')
    # axbot.tick_params(axis='y', colors='gray')
    # axbot.xaxis.label.set_color('gray')

def generate_figure(data, contour, parameters:Parameters, prefix=None, comment=None, suffix=None, filetype='pdf', roi=None):
    if prefix is None:
        prefix= ''
    if suffix is None:
        suffix=''
    if comment is None:
        comment=''
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plot_image_contour(ax, data, contour, parameters, comment=comment, roi=roi)
    ax1, ax2 = plt.subplot(2, 2, 2), plt.subplot(2, 2, 4)
    plot_difference(ax1, ax2, contour, parameters)
    prefix = f'{prefix}{suffix}'
    plt.savefig(prefix + '.' + filetype, dpi=300)