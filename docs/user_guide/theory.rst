Theory
======

Introduction
------------

sladnalsd

Mesh Generation
---------------

askldaskld

Finite Element Preliminaries
----------------------------

TODO - update for Tri3 elements...

Element Type
~~~~~~~~~~~~

dlsfkldsf

Isoparametric Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sdlfkjdslf

Shape Functions
^^^^^^^^^^^^^^^

sdlkfjdsf

Cartesian Partial Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dakljfdslkf

Numerical Integration
~~~~~~~~~~~~~~~~~~~~~

akdlskfsdf

Plane-Stress Elements
---------------------

Constitutive matrix:

.. math::
    \textbf{D} =
    \begin{bmatrix}
        \lambda + 2 \mu & \lambda & 0 \\
        \lambda & \lambda + 2 \mu & 0 \\
        0 & 0 & \mu \\
    end{bmatrix}

where $\lambda$ and $\mu$ and the Lamé parameters:

.. math::
    \lambda &= \frac{E \nu}{(1 + \nu)(1-2 \nu)} \\
    \mu &= \frac{E}{2 (1 + \nu)} \\

Local stiffness matrix:

.. math::
    \textbf{k}_{\rm e} = t \int_\Omega \textbf{B}^{\rm T} \textbf{D} \textbf{B} d \, \Omega

equates to:

.. math::
    \textbf{k}_{\rm e} = \sum_{i=1}^n w_i  \textbf{B}^{\rm T} \textbf{D} \textbf{B} J_i t \\

Stiffness Matrix Assembly
-------------------------

aslkdjaskld

Boundary Conditions
-------------------

Prescribed Nodal Displacement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
    \textbf{K}[\rm{dof}, :] = 0
    \textbf{K}[\rm{dof}, \rm{dof}] = 0
    \textbf{f}[\rm{dof}] = \textbf{u}

Nodal Spring
^^^^^^^^^^^^

.. math::
    \textbf{K}[\rm{dof}, \rm{dof}] = k

Nodal Load
^^^^^^^^^^

.. math::
    \textbf{f}[\rm{dof}] = P

Post-Processing
---------------

Nodal Forces
^^^^^^^^^^^^

.. math::
    \textbf{f} = \textbf{K} \textbf{u}

Nodal Stresses
^^^^^^^^^^^^^^

Note about gauss points vs. nodal points - Felippa

.. math::
    \textbf{\sigma} = \textbf{D} \textbf{B} \textbf{u}
