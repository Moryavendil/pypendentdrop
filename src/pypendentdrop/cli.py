#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import *
import sys
import argparse
from . import logfacility


def is_valid_float(element: any) -> bool:
    if element is None:
        return False
    try:
        if float(element) < 0:
            warning('The number cannot be negative !?')
            return False
        return True
    except ValueError:
        warning(f'This is not a number: {element}')
        return False
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

    ### Getting the image
    imagefile = args.n
    logger.debug(f'Image path provided: {imagefile}')
    if imagefile is None:
        logger.info(f'No image file provided.')
        logger.info(f'You can use the -n option to specify the image you want to analyze.')
        logger.info(f'Example: pypendantdrop-cli -n image.tif)')

    import_success, img = import_image(imagefile)
    while not import_success:
        if imagefile is None:
            print('No image file provided.')
        print('Please provide a valid path for the image you want to analyze')
        imagefile:str = str(input('Image file path: '))
        logger.debug(f'Image path provided: {imagefile}')
        import_success, img = import_image(imagefile)

    ### Getting the pixel density
    px_per_mm = args.p
    logger.debug(f'Pixel density provided: {px_per_mm} px/mm')
    if px_per_mm is None:
        logger.info(f'No pixel density provided.')
        logger.info(f'You can use the -p option specify the pixel density, in px/mm.')
        logger.info(f'Example: pypendantdrop-cli -p 57.0)')

    while not(is_valid_float(px_per_mm)):
        if px_per_mm is None:
            print('No pixel density provided.')
        print('Please provide a valid pixel density for the image you want to analyze')
        px_per_mm:str = str(input('Pixel density (px/mm): '))
        logger.debug(f'Pixel density provided: {px_per_mm} px/mm')
    px_per_mm:float = float(px_per_mm)

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

    lines = detect_contourlines(img, threshold, roi=roi)
    linelengths = [len(line) for line in lines]

    logger.debug(f'Number of lines: {len(lines)}, lengths: {linelengths}')

    cnt = detect_main_contour(img, threshold, roi=roi)

    logger.debug(f'Drop contour: {cnt.shape[1]} points')

    estimated_parameters = estimate_parameters(img, cnt, px_per_mm)

    args_parameters = Parameters()
    args_parameters.set_px_density(px_per_mm)
    args_parameters.set_a_deg(args.ai)
    args_parameters.set_x_px(args.xi)
    args_parameters.set_y_px(args.yi)
    args_parameters.set_r_mm(args.ri)
    args_parameters.set_l_mm(args.li)
    args_parameters.describe(printfn=trace, descriptor='from arguments')

    initial_parameters = Parameters()
    initial_parameters.set_px_density(px_per_mm)
    initial_parameters.set_a_deg(args.ai or estimated_parameters.get_a_deg())
    initial_parameters.set_x_px(args.xi or estimated_parameters.get_x_px())
    initial_parameters.set_y_px(args.yi or estimated_parameters.get_y_px())
    initial_parameters.set_r_mm(args.ri or estimated_parameters.get_r_mm())
    initial_parameters.set_l_mm(args.li or estimated_parameters.get_l_mm())
    initial_parameters.describe(printfn=debug, descriptor='initial')

    logger.debug(f'chi2: {compute_gap_dimensionless(cnt, parameters=initial_parameters)}')


    if args.o is not None:
        from . import plot

        plot.generate_figure(img, cnt, parameters=initial_parameters,
                             prefix=args.o, comment='estimated parameters', suffix='_initialestimate', filetype='pdf', roi=roi)
    else:
        pass
        # info('You did not ask for graphs to be generated')

    to_fit = [args.af, args.xf, args.yf, args.rf, args.lf]

    logger.debug(f'to_fit: {to_fit}')

    opti_success, opti_params = optimize_profile(cnt, parameters_initialguess=initial_parameters, to_fit=to_fit,
                                                 method=None)

    if not opti_success:
        logger.warning('Optimization failed :( Falling back to the estimated parameters.')
        sys.exit(-2)
    else:
        opti_params.describe(printfn=info, descriptor='optimized')

        logger.debug(f'chi2: {compute_gap_dimensionless(cnt, parameters=opti_params)}')

        print(f'Capillary length: {round(opti_params.get_l_mm(), 3)} mm')
        print(f'Bond number: {round(opti_params.get_bond(), 3)}')

        ### Getting the contrast density
        rhog = args.g
        logger.debug(f'Density contrast provided: {rhog} px/mm')
        if rhog is None:
            logger.info(f'No density contrast provided.')
            logger.info(f'You can use the -g option specify the density contrast (density difference times g).')
            logger.info(f'Example (for water): pypendantdrop-cli -g 9.81)')

        while not(is_valid_float(rhog)):
            if rhog is None:
                print('No density contrast provided.')
            print('Please provide a valid density contrast for the image you want to analyze')
            rhog:str = str(input('Density contrast: '))
            logger.debug(f'Density contrast provided: {rhog}')
        rhog:float = float(rhog)

        opti_params.set_densitycontrast(rhog)
        print(f'Surface tension gamma: {round(opti_params.get_surface_tension_mN(), 3)} mN/m')

        if args.o is not None:
            from . import plot
            plot.generate_figure(img, cnt, parameters=opti_params,
                                 prefix=args.o, comment='optimized parameters', suffix='_optimalestimate', filetype='pdf', roi=roi)

    sys.exit(0)