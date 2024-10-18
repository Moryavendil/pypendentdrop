#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

from ppd import error, warning, info, debug, trace, set_verbose
from ppd import anal

testdata_filepath = './assets/test_data/water_dsc1884.tif'
testdata_pxldensity = str(57.0)
testdata_rhog = str(9.81)

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

args = parser.parse_args()

if __name__ == "__main__":
    set_verbose(args.v)

    imagefile = args.n
    if imagefile is None:
        error(f'No image file provided.')
        error(f'Use -n to specify the image you want to analyze (e.g. -n {testdata_filepath})')
        error(f'Use -p to specify the pixel density, in mm/px (e.g. -p {testdata_pxldensity})')
        error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
        sys.exit(101)

    debug(f'Image path provided: {imagefile}')

    px_per_mm = args.p
    if px_per_mm is None:
        error(f'No pixel density provided.')
        error(f'Use -p to specify the pixel density, in mm/px (e.g. -p {testdata_pxldensity})')
        error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
        sys.exit(102)

    debug(f'Pixel density provided: {px_per_mm} px/mm')

    import_success, img = anal.import_image(imagefile)

    if import_success:
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

    opti_success, opti_params = anal.optimize_profile(cnt, px_per_mm=px_per_mm, parameters_initialguess=init_params, to_fit=to_fit)

    if opti_success:
        anal.talk_params(opti_params, px_per_mm, talkfn=print, name='Optimized')

        debug(f'chi2: {anal.compare_profiles(opti_params, cnt, px_per_mm=px_per_mm)}')
    else:
        warning('Optimization failed :( Falling back to the estimated parameters.')

    r0_mm = opti_params[3]
    caplength_mm = opti_params[4]

    bond = (r0_mm / caplength_mm)**2

    print(f'Bond number: {round(bond, 3)}')

    rhog = args.g
    if rhog is None:
        error(f'No density contrast provided, could not compute surface tension.')
        error(f'Use -g to specify the density contrast times gravity (e.g. -g {testdata_rhog})')
    else:
        gamma = rhog * caplength_mm**2
        print(f'Surface tension gamma: {round(gamma, 3)} mN/m')

    if args.o is not None:
        from ppd import plotresults

        plotresults.generate_figure(img, cnt, px_per_mm, init_params,
                                    prefix=args.o, comment='estimated parameters', suffix='_initialestimate', filetype='pdf')
        if opti_success:
            plotresults.generate_figure(img, cnt, px_per_mm, opti_params,
                                        prefix=args.o, comment='optimized parameters', suffix='_optimalestimate', filetype='pdf')

    sys.exit(0)
