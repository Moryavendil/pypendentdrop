import argparse

def generate_parser(program_name:str=None):
    program_name = program_name or ""
    parser = argparse.ArgumentParser(
        prog='ppd-cli',
        description=f'PyPendentDrop - {program_name}',
        epilog=f'', add_help=True)
    parser.add_argument('-n', metavar='FILENAME', help='filename', type=argparse.FileType('rb'))
    parser.add_argument('-p', metavar='PXL_DENSITY', help='Pixel density (mm/px)', type=float)
    parser.add_argument('-g', metavar='GRAVITY', help='Acceleration of gravity (typically 9.81)', type=float)
    parser.add_argument('-d', metavar='DELTARHO', help='Density contrast, in kg/L (typically 1.00 for water/air)', type=float)
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

    return parser