# PyPendentDrop

Python library (with graphical and command line interfaces) to measure surface tension from images of pendent drops.

* Package on [PyPI](https://pypi.org/project/pypendentdrop/)
* Documentation on [ReadTheDocs](https://pypendentdrop.readthedocs.io)
* Source code on  [GitHub](https://github.com/Moryavendil/pypendentdrop)


## Installation

Simply use

    pip install pypendentdrop[full]

or, if you only want to use the command-line version (resp. the graphical version), you can replace `[full]` by `[cli]` (resp. `[gui]`). Use no option to download a minimal working version of the library.

## Using PyPendentDrop

### Graphical interface

To launch the gui version, use the command

    ppt-gui

Use the relevant fields to provide an image, the pixel density of your image and the relevant physical parameters (density contrast, acceleration of gravity). Buttons allow you to estimate the parameters coarsely and to optimize this estimation. You can manually change and/or fix the values of the parameters.

### Command-line

To use the command-line version, use

    ppt-cli

Use the `-h` (help) option to list the availables options and the `-v` (verbose) option to display more information as the program goes.

### In a python script

In the import section of your script, write

    import pypendentdrop as ppd

and you can then use the functions defined in the library. An example script `examplescript.py` is provided on the GitHub repository. 

<!-- ## How it works

### The pendent drop method

[...] see scientific litterature

### PyPendentDrop

The main steps of measuring the surface tension of a liquid using the pendent drop method are

1. Select an image (if possible a high quality, high contrast image of a symmetric drop) using `ppd.import_image(filename)`

    *Optionally:* select the Region Of Interest in your image

2. Choose a threshold for your image (or use `ppd.best_threshold(image)` to find it for you)

3. Detect the contour of the drop using `ppd.detect_main_contour(image, threshold)`

4. Specify the pixel density (or pixel size) of the image

5. Obtain a coarse estimation of the parameters of the drop (tip position, angle of gravity, radius at apex, capillar length of liquid) using `ppd.estimate_parameters(image, contour, pixeldensity)`

    *Optionally:* set some of the parameters yourself if the automatically-estimated parameters are not accurate enough

6. Fit the drop profile using the estimated parameters as initial condition using `ppd.optimize_profile(contour, estimated_parameters)`

7. Knowing the density contrast (density difference between the fluids times gravity acceleration), compute the surface tension. -->

## Contact

Contact me at `pypendentdrop@protonmail.com`