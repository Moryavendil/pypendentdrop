#!venv-ppd/bin/python3.10
# -*- coding: utf-8 -*-

import sys
import argparse

from ppd import error, warning, info, debug, trace, set_verbose
from ppd import anal

parser = argparse.ArgumentParser(
    prog='ppd_commandLine',
    description='PyPendentDrop - Command line version',
    epilog='To test this, type "./ppd_commandline.py -n ./assets/test_data/water_dsc1884.tif -p 57.0 -g 9.81 -o test_drop"', add_help=True)
parser.add_argument('-n', metavar='FILENAME', help='filename', type=argparse.FileType('rb'))
parser.add_argument('-p', metavar='PXL_DENSITY', help='Pixel density (mm/px)', type=float)
parser.add_argument('-g', metavar='RHOG', help='Value of rho*g/1000 (typically 9.81)', type=float)
parser.add_argument('-o', metavar='OUTPUTFILE', help='Generate graphs [optional:file prefix]', type=str, nargs='?', const='drop', default = None)
parser.add_argument('-v', help='Verbosity (-v: info, -vv: debug, -vvv: trace)', action="count", default=0)

group1 = parser.add_argument_group('Drop contour detection options')
group1.add_argument('-t', metavar='THRESHOLD', help='Threshold level', type=int)
group1.add_argument('--tlx', help='x position of the top-left corner of the ROI', type=int)
group1.add_argument('--tly', help='y position of the top-left corner of the ROI', type=int)
group1.add_argument('--brx', help='x position of the bottom-right corner of the ROI', type=int)
group1.add_argument('--bry', help='y position of the bottom-right corner of the ROI', type=int)

group2 = parser.add_argument_group(title='Initial estimation of the parameters',
                                  description='Values of the parameters passed as initial estimation to the optimizer')
group2.add_argument('--ai', metavar='ANGLE_INIT', help='Angle of gravity (in deg)', type=float)
group2.add_argument('--xi', metavar='TIP_X_INIT', help='Tip x position (in px)', type=float)
group2.add_argument('--yi', metavar='TIP_Y_INIT', help='Tip y position (in px)', type=float)
group2.add_argument('--ri', metavar='R0_INIT', help='Drop radius r0 (in mm)', type=float)
group2.add_argument('--li', metavar='LCAP_INIT', help='Capillary length lc (in mm)', type=float)

group3 = parser.add_argument_group('Imposed parameters',
                                  description='Non-free parameters imposed to the optimizer (these are not varied to optimize the fit)')
group3.add_argument('--af', help='Fix the angle of gravity', action='store_false')
group3.add_argument('--xf', help='Fix the tip x position', action='store_false')
group3.add_argument('--yf', help='Fix the tip y position', action='store_false')
group3.add_argument('--rf', help='Fix the drop radius', action='store_false')
group3.add_argument('--lf', help='Fix the capillary length', action='store_false')

args = parser.parse_args()

if __name__ == "__main__":
    set_verbose(args.v)

    imagefile = args.n
    if imagefile is None:
        error(f'No image file provided. Please provide the image file you want to analyze using the `-n` option')
        sys.exit(101)

    debug(f'Image path provided: {imagefile}')

    px_per_mm = args.p
    if px_per_mm is None:
        error(f'No pixel density provided. PLease provide the number of px per mm using the `-p` option')
        sys.exit(102)

    debug(f'Pixel density provided: {px_per_mm} px/mm')

    success, img = anal.import_image(imagefile)

    if success:
        debug(f'Import image successful.')
    else:
        error(f'Could not retreive the image at {imagefile}')
        sys.exit(200)

    height, width = img.shape
    debug(f'Image shape: {height}x{width}')

    roi = anal.format_roi(img, [args.tlx, args.tly, args.brx, args.bry])
    debug(f'roi = {roi}')


    threshold = args.t
    if threshold is None:
        debug('Threshold not provided, using best_threshold to provide it.')
        threshold = anal.best_threshold(img, roi=roi)

    debug(f'Threshold level: {threshold}')

    lines = anal.find_contourLines(img, threshold, roi=roi)
    linelengths = [len(line) for line in lines]

    debug(f'Number of lines: {len(lines)}, lengths: {linelengths}')

    cnt = anal.find_mainContour(img, threshold, roi=roi)

    debug(f'Drop contour: {cnt.shape[1]} points')

    init_params_estimated = anal.estimate_parameters(anal.image_centre(img), cnt, px_per_mm)

    init_params_from_args = [args.ai, args.xi, args.yi, args.ri, args.li]
    anal.talk_params(init_params_from_args, px_per_mm, talkfn=trace, name='Initial (from arguments)')

    init_params = []

    for i in range(len(init_params_estimated)):
        init_params.append(init_params_from_args[i] or init_params_estimated[i]) # a or b if a is None

    anal.talk_params(init_params, px_per_mm, talkfn=info, name='Initial')

    debug(f'chi2: {anal.compare_profiles(init_params, cnt, px_per_mm=px_per_mm)}')

    to_fit = [args.af, args.xf, args.yf, args.rf, args.lf]

    debug(f'to_fit: {to_fit}')

    opti_params = anal.optimize_profile(cnt, px_per_mm=px_per_mm, parameters_initialguess=init_params, to_fit=to_fit)

    anal.talk_params(opti_params, px_per_mm, talkfn=print, name='Optimized')

    debug(f'chi2: {anal.compare_profiles(opti_params, cnt, px_per_mm=px_per_mm)}')

    r0_mm = opti_params[3]
    caplength_mm = opti_params[4]

    bond = (r0_mm / caplength_mm)**2

    print(f'Bond number: {round(bond, 3)}')

    rhog = args.g
    if rhog is None:
        error(f'No density contrast provided, could not compute surface tension. Please provide the density contrast (Delta rho*g) using the `-g` option')
    else:
        gamma = rhog * caplength_mm**2
        print(f'Surface tension gamma: {round(gamma, 3)} mN/m')

    if args.o is not None:
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import trapezoid


        plt.rcParams["figure.figsize"] = (12, 8)
        # plt.rcParams["figure.max_open_warning"] = 50

        # plt.rcParams['pgf.texsystem'] = 'pdflatex' # use this if you have LaTeX
        plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                             'figure.titlesize' : 10,
                             'axes.labelsize': 10,'axes.titlesize': 12,
                             'legend.fontsize': 10})

        def plot_image_contour(ax, image:np.ndarray, contour:np.ndarray, px_per_mm:float, fitparams:anal.Fitparams, comment=''):

            ax.set_title(f'Drop image and contour ({comment})')
            ax.imshow(image, cmap='gray')

            xcontour, ycontour = contour[0], contour[1]
            ax.plot(xcontour, ycontour, c='lime', lw=2, label='Detected contour')

            gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = fitparams

            l = max(image.shape)
            ax.plot((x_tip_position + l * np.sin(-gravity_angle), x_tip_position - l * np.sin(-gravity_angle)), (y_tip_position - l * np.cos(-gravity_angle), y_tip_position + l * np.cos(-gravity_angle)),
                    color='b', lw=2, ls='--', label=f'Direction of gravity ({comment})')

            drop_center_x = x_tip_position + r0_mm * px_per_mm * np.sin(-gravity_angle)
            drop_center_y = y_tip_position - r0_mm * px_per_mm * np.cos(-gravity_angle)
            # e1 = patches.Arc((drop_center_x, drop_center_y), 2 * r0_mm * px_per_mm, 2 * r0_mm * px_per_mm,  # WARNING CONVENTION
            #                  theta1 = 0 - gravity_angle*180/np.pi, theta2 = 180 - gravity_angle*180/np.pi,
            #                  linewidth=2, fill=False, zorder=2, color='darkred', ls='--', label=f'curvature ({comment})')
            # ax.add_patch(e1)

            Rd, Zd = anal.integrated_contour(px_per_mm, fitparams)

            ax.scatter(x_tip_position, y_tip_position, s=50, fc='k', ec='lime', linewidths=2, label=f'Tip position ({comment})', zorder=4)

            ax.plot(Rd, Zd, c='r', lw=2, label=f'Computed profile ({comment})')

            ax.legend()
            ax.set_xlabel('x [px]')
            ax.set_xlim(0, image.shape[1])
            ax.set_ylabel('y [px]')
            ax.set_ylim(image.shape[0], 0)

        def plot_difference(axtop, axbot, contour, px_per_mm, fitparams:anal.Fitparams, comment=''):
            # axtop.set_title(f'chi2: {anal.compare_profiles(fitparams, contour, px_per_mm=px_per_mm)}')
            axtop.set_title(f'Comparison of detected contour and computed profile')

            gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = fitparams

            tipRadius = r0_mm / capillary_length_mm

            # hence the profile
            R, Z = anal.compute_nondimensional_profile(tipRadius)

            # FOR COMPUTE THE DIFF : we take it backward
            XY = contour.copy()

            #moving
            XY[0] -= x_tip_position
            XY[1] -= y_tip_position

            #  rotating and scaling
            XY = anal.rotate_and_scale(XY, angle=-gravity_angle, scalefactor=-1 / (capillary_length_mm * px_per_mm))

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


        plt.figure()
        ax = plt.subplot(1, 2, 1)
        plot_image_contour(ax, img, cnt, px_per_mm, init_params, 'estimated parameters')
        ax1, ax2 = plt.subplot(2, 2, 2), plt.subplot(2, 2, 4)
        plot_difference(ax1, ax2, cnt, px_per_mm, init_params)
        name = f'{args.o}_initialestimate'
        plt.savefig(name + '.pdf', dpi=300)

        plt.figure()
        ax = plt.subplot(1, 2, 1)
        plot_image_contour(ax, img, cnt, px_per_mm, opti_params, 'estimated parameters')
        ax1, ax2 = plt.subplot(2, 2, 2), plt.subplot(2, 2, 4)
        plot_difference(ax1, ax2, cnt, px_per_mm, opti_params)
        name = f'{args.o}_optimalestimate'
        plt.savefig(name + '.pdf', dpi=300)

        # fig , (ax1, ax2) = plt.subplots(1, 2)
        # plot_image_contour(ax1, img, cnt, px_per_mm, opti_params, 'optimized parameters')
        # plot_difference(ax2, cnt, px_per_mm, opti_params)
        # name = f'{args.o}_optimalestimate'
        # plt.savefig(name + '.png', dpi=300)


    sys.exit(0)

    # fig2 , (ax21, ax22) = plt.subplots(1, 2)
    # plot_image_contour(ax21, img, cnt, px_per_mm, opti_params, 'fitted parameters')
    # plot_difference(ax22, cnt, px_per_mm, opti_params)
