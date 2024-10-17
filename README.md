# pypendentdrop
Python scripts (GUI and/or command line) to measure surface tension from images of pendent drops.

To compile the ui file

    pyside6-uic ui/ppd_mainwindow.ui -o ppd/ppd_mainwindow.py


To have the right python

    python3.10 -m venv venv-ppd
    source venv-ppd/bin/activate
    pip install -r requirements.txt


Main dep

numpy (algebra)
scipy (minimization)
contourpy (for the contour detection)
PySide6 (Qt6 for UI)
pyqtraph (responsive fast graphical UI)