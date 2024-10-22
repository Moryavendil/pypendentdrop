# PyPendentDrop

Python scripts (GUI and/or command line) to measure surface tension from images of pendent drops.

Install with

    pip install pypendentdrop


### Dependencies

All versions of PypendentDrop rely on

* `numpy` (for algebra)
* `pillow` (for image reading)
* `contourpy` (for contour detection)
* `scipy` (for parameters optimization)

Additionnaly, the GUI version relies on

* `pyqtgraph` (fast responsive graphs)
* Any Qt distribution for Python supported by PyQtGraph: `PyQt6`, `PySide6`, `PyQt5` or `PySide2`

The command-line version does not require Qt but relies on `matplotlib` for plotting the results when using the `-o` option.

### Use the command-line version

    python -m pypendentdrop.gui

Use the `-h` option to list the availables options. If you use the `-o` option (graph generation), ensure that you have matplotlib installed.

### Use the GUI version

    python -m pypendentdrop.gui

### Without installing

    python3.10 -m src.pypendentdrop.gui
    
    python3.10 -m src.pypendentdrop.tests.findsymmetryangle

if needed: contact me at: `pypendentdrop@protonmail.com`