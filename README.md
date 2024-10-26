# PyPendentDrop

Python library (with graphical and command line interfaces) to measure surface tension from images of pendent drops.

* On PyPI: [https://pypi.org/project/pypendentdrop/](https://pypi.org/project/pypendentdrop/)
* Source code: [https://github.com/Moryavendil/pypendentdrop](https://github.com/Moryavendil/pypendentdrop)


## Installation

Simply use

    pip install pypendentdrop[full]

or, if you only want to use the command-line version (resp. the graphical version), you can replace `[full]` by `[cli]` (resp. `[gui]`). Use no option to download a minimal working version of the library.

## Use PyPendentDrop

### Graphical interface

To launch the gui version, use the command

    ppt-gui

or, alternatively, 

    python -m pypendentdrop.gui

### Command-line

To use the command-line version, use

    ppt-cli

or, alternatively, 

    python -m pypendentdrop

Use the `-h` option to list the availables options. If you use the `-o` option (graph generation), ensure that you have matplotlib installed.

### In a python script

In the import section, write

    import pypendentdrop as ppd

and you can then use the functions defined in the library. An example script `examplescript.py` is provided on the GitHub repository. 

## How it works

### The pendent drop method

[...] see scientific litterature

### Using PyPendentDrop

The main steps of measuring the surface tension of a liquid using the pendent drop method are

1. Select an image (if possible a high quality, high contrast image of a symmetric drop) using `ppd.import_image(filename)`

    *Optionally:* select the Region Of Interest in your image

2. Choose a threshold for your image (or use `ppd.best_threshold(image)` to find it for you)

3. Detect the contour of the drop using `ppd.detect_main_contour(image, threshold)`

4. Specify the pixel density (or pixel size) of the image

5. Obtain a coarse estimation of the parameters of the drop (tip position, angle of gravity, radius at apex, capillar length of liquid) using `ppd.estimate_parameters(image, contour, pixeldensity)`

    *Optionally:* set some of the parameters yourself if the automatically-estimated parameters are not accurate enough

6. Fit the drop profile using the estimated parameters as initial condition using `ppd.optimize_profile(contour, estimated_parameters)`

7. Knowing the density contrast (density difference between the fluids times gravity acceleration), compute the surface tension.

<!--
## Dependencies

* The PypendentDrop library rely on

 * `numpy` (for algebra)
 * `pillow` (for image reading via `Image.open()`)
 * `contourpy` (for contour detection via `ContourGenerator.lines()`)
 * `scipy` (for parameters optimization via `minimize`)

* The graphical application depends on

 * `pyqtgraph` (fast responsive graphs)
 * Any Qt distribution for Python supported by PyQtGraph: `PyQt6`, `PySide6`, `PyQt5` or `PySide2` (when using the `[gui]` option, `PyQt5` is installed) 
 
* The command-line application depends on
 * `matplotlib`, when requesting graphs to be plotted (see `-o` option for `ppt-cli`)

to test the module, you can run (from the main directory)

    python -m pypendentdrop -n src/pypendentdrop/tests/testdata/water_2.tif -p 57 -g 9.81 -v -->

## Contact

if needed: contact me at `pypendentdrop@protonmail.com`