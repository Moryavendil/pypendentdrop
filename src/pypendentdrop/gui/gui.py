#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import pyqtgraph as pg
from .mainwindow import ppd_mainwindow

from .. import logfacility

def main():
    parser = argparse.ArgumentParser(
        prog='ppd_commandLine',
        description='PyPendentDrop - GUI line version',
        epilog=f'', add_help=True)
    parser.add_argument('-v', help='Verbosity (-v: info, -vv: logger.debug, -vvv: trace)', action="count", default=0)

    args = parser.parse_args()

    logfacility.set_verbose(args.v)

    app = pg.mkQApp("PyPendentDrop")

    mainwindow = ppd_mainwindow()

    mainwindow.show()

    sys.exit(pg.exec())
