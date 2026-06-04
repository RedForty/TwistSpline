"""
Loadable Maya plugin for the pure-Python TwistSpline (Phase 1).

Registers, with type names/ids distinct from the C++ plugin so both can be
loaded at once:
    pyTwistSplineData  (MPxData)   -- carries the spline between nodes
    pyTwistSpline      (locator)   -- builds the spline from per-CV inputs
    + a Viewport 2.0 draw override for pyTwistSpline

Load it from the Script Editor by full path::

    import maya.cmds as cmds
    cmds.loadPlugin("D:/Dropbox/scripts/TwistSpline/python/pyTwistSplinePlugin.py")

(Unload first if iterating: cmds.unloadPlugin("pyTwistSplinePlugin").)
"""

import os
import sys

import maya.api.OpenMaya as om
import maya.api.OpenMayaRender as omr


def maya_useNewAPI():
    pass


# Make the sibling ``twistspline`` package importable regardless of where the
# plugin file is loaded from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from twistspline.maya.spline_data import TwistSplineData
from twistspline.maya.spline_node import TwistSplineNode
from twistspline.maya.draw import TwistSplineDrawOverride

_VENDOR = "TwistSplinePy"
_VERSION = "0.1"


def initializePlugin(mobject):
    plugin = om.MFnPlugin(mobject, _VENDOR, _VERSION)

    plugin.registerData(
        TwistSplineData.kName, TwistSplineData.kId, TwistSplineData.creator)

    plugin.registerNode(
        TwistSplineNode.kName,
        TwistSplineNode.kId,
        TwistSplineNode.creator,
        TwistSplineNode.initialize,
        om.MPxNode.kLocatorNode,
        TwistSplineNode.kDrawClassification,
    )

    omr.MDrawRegistry.registerDrawOverrideCreator(
        TwistSplineNode.kDrawClassification,
        TwistSplineNode.kDrawRegistrantId,
        TwistSplineDrawOverride.creator,
    )


def uninitializePlugin(mobject):
    plugin = om.MFnPlugin(mobject)

    omr.MDrawRegistry.deregisterDrawOverrideCreator(
        TwistSplineNode.kDrawClassification,
        TwistSplineNode.kDrawRegistrantId,
    )
    plugin.deregisterNode(TwistSplineNode.kId)
    plugin.deregisterData(TwistSplineData.kId)
