"""
Rung 3 verification: SEE the Python kernel's frames in the Maya viewport.

This is the most intuitive check -- it draws the spline curve plus a little
RGB axis tripod (tan=red, norm=green, binorm=blue) at evenly spaced samples,
so you can watch the twist distribute evenly with no pops, exactly like the
TwistSpline_TwistDemo.gif but driven entirely by the pure-Python kernel.

It uses ``maya.cmds`` only for *drawing* -- the math still comes from the
dependency-free ``twistspline`` package.

Usage (in the Maya Script Editor)::

    import sys
    sys.path.append("/path/to/TwistSpline/python")   # folder containing twistspline/
    import viz_maya
    viz_maya.demo_pure()      # even, flip-free RMF (no twist)
    viz_maya.demo_twist()     # 360 deg of user twist, distributed by length
"""

import os
import sys
from math import radians, cos, sin

# Make the sibling ``twistspline`` package importable when this file is run
# directly from the Script Editor.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = None
if _HERE and _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import maya.cmds as cmds

from twistspline.core import make_spline


S_CURVE = [
    [0.0, 0.0, 0.0],
    [3.0, 2.0, 1.0],
    [6.0, 0.0, 3.0],
    [9.0, -2.0, 1.0],
    [12.0, 0.0, 0.0],
]


def _colored_line(a, b, rgb, parent):
    crv = cmds.curve(degree=1, point=[list(a), list(b)])
    shp = cmds.listRelatives(crv, shapes=True)[0]
    cmds.setAttr(shp + ".overrideEnabled", 1)
    cmds.setAttr(shp + ".overrideRGBColors", 1)
    cmds.setAttr(shp + ".overrideColorRGB", *rgb)
    cmds.parent(crv, parent)
    return crv


def visualize(spline, samples=40, axis_scale=0.6, name="twistSplineViz"):
    """Draw a sampled TwistSpline: the curve + an RGB tripod at each sample."""
    if cmds.objExists(name):
        cmds.delete(name)
    grp = cmds.group(empty=True, name=name)

    # The curve itself (the LUT points).
    pts = spline.get_points()
    cmds.parent(cmds.curve(degree=1, point=[list(p) for p in pts]), grp)

    lo, hi = spline.param_range
    for i in range(samples):
        p = lo + (hi - lo) * i / (samples - 1)
        f = spline.matrix_at_param(p)
        o = f.tran
        tip_t = [o[k] + f.tan[k] * axis_scale for k in range(3)]
        tip_n = [o[k] + f.norm[k] * axis_scale for k in range(3)]
        tip_b = [o[k] + f.binorm[k] * axis_scale for k in range(3)]
        _colored_line(o, tip_t, (1.0, 0.1, 0.1), grp)   # tangent  = red
        _colored_line(o, tip_n, (0.1, 1.0, 0.1), grp)   # normal   = green
        _colored_line(o, tip_b, (0.1, 0.3, 1.0), grp)   # binormal = blue
    return grp


def demo_pure():
    """Even, flip-free RMF with no twist (the 'magic by default' case)."""
    spline = make_spline(S_CURVE)
    return visualize(spline, name="twistSplineViz_pure")


def demo_twist(turns=1.0):
    """Apply ``turns`` full rotations of user twist, pinned start->end, so you
    can watch the green/blue axes rotate evenly along the length."""
    n = len(S_CURVE)
    user_twists = [0.0] * n
    user_twists[-1] = radians(360.0 * turns)
    twist_locks = [0.0] * n
    twist_locks[0] = 1.0
    twist_locks[-1] = 1.0
    spline = make_spline(S_CURVE, user_twists=user_twists, twist_locks=twist_locks)
    return visualize(spline, name="twistSplineViz_twist")
