.. PyPendentDrop documentation master file, created by
   sphinx-quickstart on Wed Oct 23 22:18:20 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============

You can install PyPendentDrop and all its dependencies using ``pip``:

.. code-block:: console

    $ pip install pypendentdrop[full]


If you are interested in only some of the functionalities of PyPendentDrop and/or do not want to download some packages, see the :ref:`Installation options<installation_options>` below.

.. _installation_options:

Installation options
--------------------

We recommend using the ``[full]`` option when installing PyPendentDrop. Other installation options available are:

Minimal (no option)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ pip install pypendentdrop

This will install the PyPendentDrop API, along with its core dependencies
 * ``numpy`` (for algebra)
 * ``pillow`` (for image reading via ``Image.open()``)
 * ``contourpy >= 1.0`` (for contour detection via ``ContourGenerator.lines()``)
 * ``scipy >= 1.7`` (for parameters optimization via ``minimize``)


GUI only
~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ pip install pypendentdrop[gui]

This will install the PyPendentDrop API, along with its core dependencies and
 * ``pyqtgraph`` (for fast responsive interactive graphs)
 * ``PyQt5`` (to have Qt)

If you prefer using another Python binding for Qt, you can use any of the ones supported by ``pyqtgraph``. See the `pyqtgraph documentation <https://pyqtgraph.readthedocs.io/en/latest/getting_started/how_to_use.html#pyqt-and-pyside>`_ to know which python wrappers for Qt are supported.

CLI only
~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ pip install pypendentdrop[cli]

This will install the PyPendentDrop API, along with its core dependencies and
 * ``matplotlib`` (for clean graphs and the ability to save them)


Installation throubleshooting
-----------------------------

I dont have pip 
~~~~~~~~~~~~~~~~~

The Python package manager, ``pip``, is most likely installed by default (if not, you should probably `install it <https://pip.pypa.io/en/stable/installation/>`_). However, you may need to upgrade pip to the lasted version:

.. code-block:: console

    $ pip install --upgrade pip


My packages and PendentDrop's dependencies are incompatible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should use a `virtual environment <https://docs.python.org/3/library/venv.html>`_.

.. code-block:: console

    $ python -m venv venv/ppd
    $ source venv/ppd/bin/activate

..
    TextWithLinkToFn <--> :func:`TextWithLinkToFn <pypendentdrop.import_image>`


