"""Validation tests taken from the Strand7 validation manual.

Insert reference here...
"""


def test_vls1():
    """VLS1: Elliptic Membrane.

    An elliptical plate with an elliptical hole is analysed.
    Outer ellipse - (x / 3.25)**2 + (y / 2.75)**2 = 1
    Inner ellipse - (x / 2)**2 + (y)**2 = 1

    Uniform outward pressure is applied at the outer boundary. (10 MPa)
    As both the structure and the loading condition are symmetric, only a quarter of the
    structure is modelled. (1st quadrant).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=2, y=0) of 92.7 MPa.

    TODO - must first implement line load normal to curve.
    """
    pass


def test_vls8():
    """VLS8: Circular Membrane - Edge Pressure.

    A ring under uniform external pressure of 100 MPa is analysed.
    (inner = 10m, outer = 11m, t=1m).

    One eighth of the ring (45 deg) is modelled via nodal restraints in a UCS.
    In the validation of plane-stress one-quarter of the model will be analysed (no
    implementation of 45 deg rollers).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=10, y=0) of -1150 MPa.

    TODO - must first implement line load normal to curve.
    """
    pass


def test_vls9():
    """VLS9: Circular Membrane - Point Load.

    A ring under concentrated forces is analysed. (10000 kN at 4 x 45 deg).
    (inner = 10m, outer = 11m, t=1m).

    One eighth of the ring (45 deg) is modelled via nodal restraints in a UCS.
    In the validation of plane-stress one-quarter of the model will be analysed (no
    implementation of 45 deg rollers).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=10, y=0) of -53.2 MPa.
    """
    pass
