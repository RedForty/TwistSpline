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


def _ensure_on_path(plugin):
    """Make the sibling ``twistspline`` package importable.

    Maya executes plugin files without a reliable ``__file__``, so derive the
    plugin's own directory from ``MFnPlugin.loadPath()`` (the dir it was loaded
    from) and put it on ``sys.path``.
    """
    here = None
    try:
        here = plugin.loadPath()
    except Exception:
        here = None
    if not here:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            here = None
    if here and here not in sys.path:
        sys.path.insert(0, here)


def initializePlugin(mobject):
    plugin = om.MFnPlugin(mobject, _VENDOR, _VERSION)
    _ensure_on_path(plugin)
    from twistspline.maya.spline_data import TwistSplineData
    from twistspline.maya.spline_node import TwistSplineNode
    from twistspline.maya.draw import TwistSplineDrawOverride
    from twistspline.maya.rider_node import RiderConstraint
    from twistspline.maya.tangent_node import TwistTangentNode
    from twistspline.maya.tangent_multi_node import TwistMultiTangentNode

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

    plugin.registerNode(
        RiderConstraint.kName,
        RiderConstraint.kId,
        RiderConstraint.creator,
        RiderConstraint.initialize,
        om.MPxNode.kDependNode,
    )

    plugin.registerNode(
        TwistTangentNode.kName,
        TwistTangentNode.kId,
        TwistTangentNode.creator,
        TwistTangentNode.initialize,
        om.MPxNode.kDependNode,
    )

    plugin.registerNode(
        TwistMultiTangentNode.kName,
        TwistMultiTangentNode.kId,
        TwistMultiTangentNode.creator,
        TwistMultiTangentNode.initialize,
        om.MPxNode.kDependNode,
    )


def uninitializePlugin(mobject):
    plugin = om.MFnPlugin(mobject)
    _ensure_on_path(plugin)
    from twistspline.maya.spline_node import TwistSplineNode
    from twistspline.maya.spline_data import TwistSplineData
    from twistspline.maya.draw import TwistSplineDrawOverride
    from twistspline.maya.rider_node import RiderConstraint
    from twistspline.maya.tangent_node import TwistTangentNode
    from twistspline.maya.tangent_multi_node import TwistMultiTangentNode

    plugin.deregisterNode(TwistMultiTangentNode.kId)
    plugin.deregisterNode(TwistTangentNode.kId)
    plugin.deregisterNode(RiderConstraint.kId)
    omr.MDrawRegistry.deregisterDrawOverrideCreator(
        TwistSplineNode.kDrawClassification,
        TwistSplineNode.kDrawRegistrantId,
    )
    plugin.deregisterNode(TwistSplineNode.kId)
    plugin.deregisterData(TwistSplineData.kId)
