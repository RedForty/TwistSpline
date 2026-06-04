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


_VENDOR = "TwistSplinePy"
_VERSION = "0.1"


def _ensure_on_path():
    """Make the sibling ``twistspline`` package importable.

    Maya executes plugin files without a reliable ``__file__``, so fall back to
    locating this module via the loaded-plugin list, then to MGlobal's plugin
    search. If none resolve, the caller is expected to have added the ``python``
    dir to ``sys.path`` already.
    """
    here = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        try:
            path = om.MFnPlugin.findPlugin("pyTwistSplinePlugin")
            if path:
                here = os.path.dirname(path)
        except Exception:
            here = None
    if here and here not in sys.path:
        sys.path.insert(0, here)


def initializePlugin(mobject):
    _ensure_on_path()
    from twistspline.maya.spline_data import TwistSplineData
    from twistspline.maya.spline_node import TwistSplineNode
    from twistspline.maya.draw import TwistSplineDrawOverride

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
    _ensure_on_path()
    from twistspline.maya.spline_node import TwistSplineNode
    from twistspline.maya.spline_data import TwistSplineData
    from twistspline.maya.draw import TwistSplineDrawOverride

    plugin = om.MFnPlugin(mobject)

    omr.MDrawRegistry.deregisterDrawOverrideCreator(
        TwistSplineNode.kDrawClassification,
        TwistSplineNode.kDrawRegistrantId,
    )
    plugin.deregisterNode(TwistSplineNode.kId)
    plugin.deregisterData(TwistSplineData.kId)
