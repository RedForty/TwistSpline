"""
twistspline -- a pure-Python reference port of the TwistSpline kernel.

This package re-implements the math from the C++ headers (``src/twistSpline.h``
+ ``src/twistSpline_maya.h``) with zero third-party dependencies, so it can be
run and verified anywhere -- including inside Maya's bundled Python.

Typical use::

    from twistspline.core import make_spline
    spline = make_spline([[0, 0, 0], [3, 2, 0], [6, 0, 2]])
    frame = spline.matrix_at_param(3.0)        # -> Frame(tan, norm, binorm, ...)
"""

from .core import TwistSpline, TwistSplineSegment, Frame, make_spline

__all__ = ["TwistSpline", "TwistSplineSegment", "Frame", "make_spline"]
