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

### Play with the library
If you wish to write your own Python scripts using the PyPendentDrop library, here is a minimal example script

	from ppd import anal
	
	filepath = './assets/test_data/water_dsc1884.tif'
	pxldensity = 57.0
	rhog = 9.812
	roi = [10, 90, 300, 335]
	
	success, img = anal.import_image(filepath)
	
	threshold = anal.best_threshold(img, roi=roi)
	
	cnt = anal.find_mainContour(img, threshold, roi=roi)
	
	estimated_parameters = anal.estimate_parameters(anal.image_centre(img), cnt, pxldensity)
	
	optimized_parameters = anal.optimize_profile(cnt, px_per_mm=pxldensity, parameters_initialguess=estimated_parameters)
	
	gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = optimized_parameters
	
	bond = (r0_mm / capillary_length_mm)**2
	print(f'Bond number: {round(bond, 3)}')
	
	gamma = rhog * capillary_length_mm**2
	print(f'Surface tension gamma: {round(gamma, 3)} mN/m')
	
	### Plotting a comparison between the estimated and optimized parameters
	import matplotlib.pyplot as plt
	from ppd import plotresults
	
	fig, (ax1, ax2) = plt.subplots(1, 2)
	plotresults.plot_image_contour(ax1, img, cnt, pxldensity, estimated_parameters, 'estimated', roi=roi)
	plotresults.plot_image_contour(ax2, img, cnt, pxldensity, optimized_parameters, 'optimized', roi=roi)
	
	plt.savefig('comparison.png', dpi=300)