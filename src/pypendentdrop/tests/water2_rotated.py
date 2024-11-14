import sys
import argparse

import pypendentdrop as ppd

from .. import logfacility
from ..ppd_parse import generate_parser

testdata_filepath = 'src/pypendentdrop/tests/testdata/water_2_rotated.tif'
testdata_pxldensity = 57.0
testdata_g = 9.81
testdata_d = 1.00
testdata_roi = None

parser = generate_parser('test-water2_rotated')
args = parser.parse_args(['-n', str(testdata_filepath), '-p', str(testdata_pxldensity),
                          '-g', str(testdata_g), '-d', str(testdata_d),
                          '-vvv', '-o', 'water2_rotated'])

if __name__ == "__main__":
    logfacility.set_verbose(args.v)

    imagefile = args.n
    if imagefile is None:
        ppd.logger.error(f'No image file provided.')
        ppd.logger.error(f'Use -n to specify the image you want to analyze (e.g. -n {testdata_filepath})')
        sys.exit(101)

    ppd.logger.debug(f'Image path provided: {imagefile}')

    px_per_mm = args.p
    if px_per_mm is None:
        ppd.logger.error(f'No pixel density provided.')
        ppd.logger.error(f'Use -p to specify the pixel density, in mm/px (e.g. -p {testdata_pxldensity})')
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
        ppd.logger.debug('Threshold not provided, using auto_threshold to provide it.')
        threshold = ppd.auto_threshold(img, roi=roi)

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
