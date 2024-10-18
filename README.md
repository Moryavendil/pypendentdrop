# PyPendentDrop

Python scripts (GUI and/or command line) to measure surface tension from images of pendent drops.

### Dependencies

All versions of PypendentDrop rely on

* `numpy` (for algebra)
* `contourpy` (for contour detection)
* `scipy` (for parameters optimization)

Additionnaly, the GUI version relies on

* `PySide6` (Qt6 for UI)
* `pyqtraph` (responsive fast graphical UI)

The command-line version with graph-generation on do not require Qt but relies on `matplotlib` for plotting the results.

### Run the command-line version
TODO

### Run the GUI version
You can run the GUI using a Python distribution that has all required dependencies. Use of Python 3.10(.15) in a virtual environment is recommended.

To create a suitable environment, use the following commands:

    python3.10 -m venv venv-ppd
    source venv-ppd/bin/activate
    pip install -r requirements.txt


Then if needed, compile the ui file:

    pyside6-uic ui/ppd_mainwindow.ui -o ppd/ppd_mainwindow.py

To run PyPendentDrop (GUI version), 

* If within a suitable python environment, run 

        python3.10 launchgui.py`


* if you prefer using another virtual environment, run `chmod 777 launchgui.py` once to make the file executable, and then

        ./launchgui.py

will do the trick (if you have the `venv-ppd` virtual environment set-up as shown above)
