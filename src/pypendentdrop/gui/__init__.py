### This is pasted forn PyQtGraphs
import os, sys

PYSIDE = 'PySide'
PYSIDE2 = 'PySide2'
PYSIDE6 = 'PySide6'
PYQT4 = 'PyQt4'
PYQT5 = 'PyQt5'
PYQT6 = 'PyQt6'

QT_LIB = os.getenv('PYQTGRAPH_QT_LIB')

if QT_LIB is not None:
    try:
        __import__(QT_LIB)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Environment variable PYQTGRAPH_QT_LIB is set to '{os.getenv('PYQTGRAPH_QT_LIB')}', but no module with this name was found.")

## Automatically determine which Qt package to use (unless specified by
## environment variable).
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import in the order
## specified in libOrder.
if QT_LIB is None:
    libOrder = [PYQT6, PYSIDE6, PYQT5, PYSIDE2]

    for lib in libOrder:
        if lib in sys.modules:
            QT_LIB = lib
            break

if QT_LIB is None:
    for lib in libOrder:
        qt = lib + '.QtCore'
        try:
            __import__(qt)
            QT_LIB = lib
            break
        except ImportError:
            pass

if QT_LIB is None:
    raise ImportError("PyPendentDrop GUI requires one of PyQt5, PyQt6, PySide2 or PySide6; none of these packages could be imported (try `pip install PyQt5`).")


try:
    pg = 'pyqtgraph'
    __import__(pg)
except ImportError:
    raise ImportError("PyPendentDrop GUI requires pyqtgraph; which could not be imported (try `pip install pyqtgraph`).")
