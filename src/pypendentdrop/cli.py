from . import *
import sys
import argparse
from . import logfacility

def main():
    testdata_filepath = './assets/test_data/water_dsc1884.tif'
    testdata_pxldensity = str(57.0)
    testdata_rhog = str(9.81)

    parser = argparse.ArgumentParser(
        prog='ppd_commandLine',
        description='PyPendentDrop - Command line version',
        epilog=f'', add_help=True)
    parser.add_argument('-n', metavar='FILENAME', help='filename', type=argparse.FileType('rb'))
    parser.add_argument('-p', metavar='PXL_DENSITY', help='Pixel density (mm/px)', type=float)
    parser.add_argument('-g', metavar='RHOG', help='Value of rho*g/1000 (typically 9.81)', type=float)
    parser.add_argument('-o', metavar='OUTPUTFILE', help='Generate graphs [optional:file prefix]', type=str, nargs='?', const='drop', default = None)
    parser.add_argument('-v', help='Verbosity (-v: info, -vv: logger.debug, -vvv: trace)', action="count", default=0)

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

    logfacility.set_verbose(args.v)

    imagefile = args.n
    if imagefile is None:
        logger.error(f'No image file provided.')
        logger.error(f'Use -n to specify the image you want to analyze (e.g. -n image.tif)')
        logger.error(f'Use -p to specify the pixel density, in mm/px (e.g. -p {testdata_pxldensity})')
        logger.error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
        sys.exit(101)

    logger.debug(f'Image path provided: {imagefile}')

    px_per_mm = args.p
    if px_per_mm is None:
        logger.error(f'No pixel density provided.')
        logger.error(f'Use -p to specify the pixel density, in mm/px (e.g. -p {testdata_pxldensity})')
        logger.error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
        sys.exit(102)

    logger.debug(f'Pixel density provided: {px_per_mm} px/mm')

    import_success, img = import_image(imagefile)

    if import_success:
        logger.debug(f'Import image successful.')
    else:
        logger.error(f'Could not retreive the image at {imagefile}')
        sys.exit(200)

    height, width = img.shape
    logger.debug(f'Image shape: {height}x{width}')

    roi = format_roi(img, [args.tlx, args.tly, args.brx, args.bry])
    logger.debug(f'roi = {roi}')


    threshold = args.t
    if threshold is None:
        logger.debug('Threshold not provided, using best_threshold to provide it.')
        threshold = best_threshold(img, roi=roi)

    logger.debug(f'Threshold level: {threshold}')

    lines = find_contourLines(img, threshold, roi=roi)
    linelengths = [len(line) for line in lines]

    logger.debug(f'Number of lines: {len(lines)}, lengths: {linelengths}')

    cnt = find_mainContour(img, threshold, roi=roi)

    logger.debug(f'Drop contour: {cnt.shape[1]} points')

    estimated_parameters = estimate_parameters(image_centre(img), cnt, px_per_mm)

    args_parameters = Parameters()
    args_parameters.set_px_density(px_per_mm)
    args_parameters.set_a_deg(args.ai)
    args_parameters.set_x_px(args.xi)
    args_parameters.set_y_px(args.yi)
    args_parameters.set_r_mm(args.ri)
    args_parameters.set_l_mm(args.li)
    args_parameters.describe(printfn=trace, name='from arguments')

    initial_parameters = Parameters()
    initial_parameters.set_px_density(px_per_mm)
    initial_parameters.set_a_deg(args.ai or estimated_parameters.get_a_deg())
    initial_parameters.set_x_px(args.xi or estimated_parameters.get_x_px())
    initial_parameters.set_y_px(args.yi or estimated_parameters.get_y_px())
    initial_parameters.set_r_mm(args.ri or estimated_parameters.get_r_mm())
    initial_parameters.set_l_mm(args.li or estimated_parameters.get_l_mm())
    initial_parameters.describe(printfn=debug, name='initial')


    ppd.logger.debug(f'chi2: {ppd.compute_gap_dimensionless(cnt, parameters=initial_parameters)}')

    to_fit = [args.af, args.xf, args.yf, args.rf, args.lf]

    logger.debug(f'to_fit: {to_fit}')

    opti_success, opti_params = optimize_profile(cnt, parameters_initialguess=initial_parameters, to_fit=to_fit,
                                                 method=None)

    if opti_success:
        opti_params.describe(printfn=info, name='optimized')

        logger.debug(f'chi2: {compute_gap_dimensionless(cnt, parameters=opti_params)}')
    else:
        logger.warning('Optimization failed :( Falling back to the estimated parameters.')


    print(f'Bond number: {round(opti_params.get_bond(), 3)}')

    rhog = args.g
    if rhog is None:
        logger.error(f'No density contrast provided, could not compute surface tension.')
        logger.error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
    else:
        opti_params.set_densitycontrast(rhog)
        print(f'Surface tension gamma: {round(opti_params.get_surface_tension(), 3)} mN/m')

    if args.o is not None:
        from . import plot

        plot.generate_figure(img, cnt, parameters=initial_parameters,
                             prefix=args.o, comment='estimated parameters', suffix='_initialestimate', filetype='pdf', roi=roi)
        if opti_success:
            plot.generate_figure(img, cnt, parameters=opti_params,
                                 prefix=args.o, comment='optimized parameters', suffix='_optimalestimate', filetype='pdf', roi=roi)

    sys.exit(0)