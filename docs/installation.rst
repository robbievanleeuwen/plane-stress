.. _label-installation:

Installation
============

These instructions will get you a copy of ``planestress`` up and running on your
machine. You will need a working copy of python 3.9, 3.10 or 3.11 to get started.

Installing ``planestress``
--------------------------------

``planestress`` uses `shapely <https://github.com/shapely/shapely>`_ to prepare
the plane-stress geometry and `gmsh <https://gmsh.info/>`_ to generate a triangular or
rectangular mesh. `numpy <https://github.com/numpy/numpy>`_ and
`scipy <https://github.com/scipy/scipy>`_ are used to aid finite element computations,
while `matplotlib <https://github.com/matplotlib/matplotlib>`_ and
`rich <https://github.com/Textualize/rich>`_ are used for post-processing. Finally,
`click <https://github.com/pallets/click>`_ is used to power the ``planestress`` CLI.

``planestress`` and all of its dependencies can be installed through the python
package index:

.. code-block:: shell

    pip install planestress
