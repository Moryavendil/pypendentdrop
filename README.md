# PyPendentDrop

Python scripts (GUI and/or command line) to measure surface tension from images of pendent drops.

### Dependencies

All versions of PypendentDrop rely on

* `numpy` (for algebra)
* `contourpy` (for contour detection)
* `scipy` (for parameters optimization)

Additionnaly, the GUI version relies on

* `PySide6` (Qt6 for UI)
* `pyqtgraph` (fast responsive graphs)

The command-line version do not require Qt but relies on `matplotlib` for plotting the results if using th `-o` option.

### Use the command-line version
Install scipy if you do not have it 

    pip install scipy

and then either 

* Run the file using python `python ppd_commandline.py`,

* or make the file executable using `chmod +x ppd_commandline.py` and then execute it `./ppd_commandline.py`.

Use the `-h` option to list the availables options. If you use the `-o` option (graph generation), ensure that you have matplotlib installed.

### Use the GUI version
You can run the GUI using a Python distribution that has all required dependencies. Use of Python 3.10(.15) in a virtual environment is recommended.

To create a suitable environment, use the following commands:

    python3.10 -m venv venv-ppd
    source venv-ppd/bin/activate
    pip install -r requirements_gui.txt


Then if needed, compile the ui file:

    pyside6-uic ui/ppd_mainwindow.ui -o ppd/ppd_mainwindow.py

To run PyPendentDrop (GUI version), 

* If within a suitable python environment, simply run `python3.10 ppd_gui.py`,

* if you have the `venv-ppd` virtual environment set-up as shown above, run `chmod +x launchgui.py` once to make the file executable, and then

        ./ppd_gui.py

will launch the GUI version.
