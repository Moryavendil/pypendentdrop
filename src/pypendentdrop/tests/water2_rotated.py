import sys
import argparse

import pypendentdrop as ppd

from .. import logfacility

testdata_filepath = 'src/pypendentdrop/tests/testdata/water_2_rotated.tif'
testdata_pxldensity = 57.0
testdata_rhog = 9.81
testdata_roi = None

parser = argparse.ArgumentParser(
    prog='ppd_commandLine',
    description='PyPendentDrop - Command line version',
    epilog=f'To test this, type "./ppd_commandline.py -n {testdata_filepath} -p {testdata_pxldensity} -g {testdata_rhog} -o test_drop"', add_help=True)
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

args = parser.parse_args(['-n', str(testdata_filepath), '-p', str(testdata_pxldensity), '-g', str(testdata_rhog),
                          '-vvv', '-o', 'water2_rotated'])

if __name__ == "__main__":
    logfacility.set_verbose(args.v)

    imagefile = args.n
    if imagefile is None:
        ppd.logger.error(f'No image file provided.')
        ppd.logger.error(f'Use -n to specify the image you want to analyze (e.g. -n {testdata_filepath})')
        ppd.logger.error(f'Use -p to specify the pixel density, in mm/px (e.g. -p {testdata_pxldensity})')
        ppd.logger.error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
        sys.exit(101)

    ppd.logger.debug(f'Image path provided: {imagefile}')

    px_per_mm = args.p
    if px_per_mm is None:
        ppd.logger.error(f'No pixel density provided.')
        ppd.logger.error(f'Use -p to specify the pixel density, in mm/px (e.g. -p {testdata_pxldensity})')
        ppd.logger.error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
        sys.exit(102)

    ppd.logger.debug(f'Pixel density provided: {px_per_mm} px/mm')

    import_success, img = ppd.import_image(imagefile)

    if import_success:
        ppd.logger.debug(f'Import image successful.')
    else:
        ppd.logger.error(f'Could not retreive the image at {imagefile}')
        sys.exit(200)

    height, width = img.shape
    ppd.logger.debug(f'Image shape: {height}x{width}')

    roi = ppd.format_roi(img, [args.tlx, args.tly, args.brx, args.bry])
    ppd.logger.debug(f'roi = {roi}')

    threshold = args.t
    if threshold is None:
        ppd.logger.debug('Threshold not provided, using best_threshold to provide it.')
        threshold = ppd.best_threshold(img, roi=roi)

    ppd.logger.debug(f'Threshold level: {threshold}')

    lines = ppd.detect_contourlines(img, threshold, roi=roi)
    linelengths = [len(line) for line in lines]

    ppd.logger.debug(f'Number of lines: {len(lines)}, lengths: {linelengths}')

    cnt = ppd.detect_main_contour(img, threshold, roi=roi)

    ppd.logger.debug(f'Drop contour: {cnt.shape[1]} points')

    estimated_parameters = ppd.estimate_parameters(img, cnt, px_per_mm)

    args_parameters = ppd.Parameters()
    args_parameters.set_px_density(px_per_mm)
    args_parameters.set_a_deg(args.ai)
    args_parameters.set_x_px(args.xi)
    args_parameters.set_y_px(args.yi)
    args_parameters.set_r_mm(args.ri)
    args_parameters.set_l_mm(args.li)
    args_parameters.describe(printfn=ppd.trace, descriptor='from arguments')

    initial_parameters = ppd.Parameters()
    initial_parameters.set_px_density(px_per_mm)
    initial_parameters.set_a_deg(args.ai or estimated_parameters.get_a_deg())
    initial_parameters.set_x_px(args.xi or estimated_parameters.get_x_px())
    initial_parameters.set_y_px(args.yi or estimated_parameters.get_y_px())
    initial_parameters.set_r_mm(args.ri or estimated_parameters.get_r_mm())
    initial_parameters.set_l_mm(args.li or estimated_parameters.get_l_mm())
    initial_parameters.describe(printfn=ppd.debug, descriptor='initial')

    ppd.logger.debug(f'chi2: {ppd.compute_gap_dimensionless(cnt, parameters=initial_parameters)}')

    to_fit = [args.af, args.xf, args.yf, args.rf, args.lf]

    ppd.logger.debug(f'to_fit: {to_fit}')

    opti_success, opti_params = ppd.optimize_profile(cnt, parameters_initialguess=initial_parameters, to_fit=to_fit,
                                                     method=None)

    if opti_success:
        opti_params.describe(printfn=ppd.info, descriptor='optimized')

        ppd.logger.debug(f'chi2: {ppd.compute_gap_dimensionless(cnt, parameters=opti_params)}')
    else:
        ppd.logger.warning('Optimization failed :( Falling back to the estimated parameters.')

    # r0_mm = opti_params[3]
    # caplength_mm = opti_params[4]
    #
    # bond = (r0_mm / caplength_mm)**2
    #
    # print(f'Bond number: {round(bond, 3)}')

    # rhog = args.g
    # if rhog is None:
    #     ppd.logger.error(f'No density contrast provided, could not compute surface tension.')
    #     ppd.logger.error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
    # else:
    #     gamma = rhog * caplength_mm**2
    #     print(f'Surface tension gamma: {round(gamma, 3)} mN/m')

    if args.o is not None:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from .. import plot

        plot.generate_figure(img, cnt, initial_parameters,
                             prefix=args.o, comment='estimated parameters', suffix='_initialestimate', filetype='pdf', roi=roi)
        if opti_success:
            plot.generate_figure(img, cnt, opti_params,
                                 prefix=args.o, comment='optimized parameters', suffix='_optimalestimate', filetype='pdf', roi=roi)
        plt.show()

    sys.exit(0)
