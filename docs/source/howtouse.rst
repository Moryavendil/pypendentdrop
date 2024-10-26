How to use
========================

Use PyPendentDrop to measure a surface tension
-------------------------------------------------

The main steps of measuring the surface tension of a liquid via PyPendentDrop are

1. Import an image (if possible a high quality, high contrast image of a symmetric drop) using ``ppd.import_image(filename)``

    *Optionally:* Specify the Region Of Interest in your image

2. Choose a threshold for your image (or use ``ppd.best_threshold(image)`` to find it for you)

3. Detect the contour of the drop using ``ppd.detect_main_contour(image, threshold)``

4. Specify the pixel density (or pixel size) of the image

5. Obtain a coarse estimation of the parameters of the drop (tip position, angle of gravity, radius at apex, capillar length of liquid) using ``ppd.estimate_parameters(image, contour, pixeldensity)``

    *Optionally:* Tune some of the parameters yourself if the automatically-estimated parameters are not accurate enough

6. Fit the drop profile using the estimated parameters as initial condition using ``ppd.optimize_profile(contour, estimated_parameters)``

7. Knowing the density contrast (density difference between the fluids times gravity acceleration), compute the surface tension.


Start PyPendentDrop
--------------------

Graphical app
~~~~~~~~~~~~~~~~~~~~

Launch the graphical app using

.. code-block:: console

   $ ppd-gui

or

.. code-block:: console

   $ python -m pypendentdrop.gui


Command-line app
~~~~~~~~~~~~~~~~~~~~

Launch the command-line application using

.. code-block:: console

   $ ppd-gui

or

.. code-block:: console

   $ python -m pypendentdrop.gui

Use the ``-h`` option to list the available options. You need to have ``matplotlib`` installed to use the ``-o`` option (graph generation).

PyPendentGraph API
~~~~~~~~~~~~~~~~~~~~

Of course, you can use all the functions provided by the PyPendentGraph library in a custom Python script, if you wish to do the analysis in your own style (different contour selection, optimization method...) or integrate PyPendentDrop in another application. In the import section of your program, simply import the library using

.. code-block:: python

    import pypendentdrop as ppd

and you can then use the functions defined in the library (see the API Reference to understand what each function does). An example script, ``examplescript.py``, using some of the PyPendentDrop functionalities, is available `in the GitHub repository <https://github.com/Moryavendil/pypendentdrop>`_.
