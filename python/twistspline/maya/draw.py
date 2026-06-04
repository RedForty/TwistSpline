"""
Viewport 2.0 draw override for the ``pyTwistSpline`` node.

Pure-Python ``MPxDrawOverride`` (the equivalent of ``src/drawOverride.cpp``).
Draws the spline as a connected line strip, and -- when ``debugDisplay`` is on
-- an RGB axis tripod (tan=red, norm=green, binorm=blue) at each LUT sample, so
you can see the rotation-minimizing frames and the twist along the length.
"""

import maya.api.OpenMaya as om
import maya.api.OpenMayaRender as omr
import maya.api.OpenMayaUI as omui

from .spline_node import TwistSplineNode


def maya_useNewAPI():
    pass


class _SplineUserData(om.MUserData):
    def __init__(self):
        try:
            super(_SplineUserData, self).__init__(False)  # deleteAfterUse
        except TypeError:
            # Newer Maya: MUserData takes no args.
            super(_SplineUserData, self).__init__()
        self.points = []          # list of MPoint, the curve
        self.frame_lines = []     # list of (MPoint a, MPoint b, MColor)
        self.draw_spline = True


def _read_spline(node):
    plug = om.MPlug(node, TwistSplineNode.aOutputSpline)
    try:
        obj = plug.asMObject()
        return om.MFnPluginData(obj).data().spline()
    except Exception:
        return None


def _read_bool(node, attr, default):
    try:
        return om.MPlug(node, attr).asBool()
    except Exception:
        return default


def _read_double(node, attr, default):
    try:
        return om.MPlug(node, attr).asDouble()
    except Exception:
        return default


class TwistSplineDrawOverride(omr.MPxDrawOverride):
    def __init__(self, obj):
        super(TwistSplineDrawOverride, self).__init__(obj, None, False)

    @staticmethod
    def creator(obj):
        return TwistSplineDrawOverride(obj)

    def supportedDrawAPIs(self):
        return omr.MRenderer.kAllDevices

    def hasUIDrawables(self):
        return True

    def prepareForDraw(self, objPath, cameraPath, frameContext, oldData):
        node = objPath.node()
        data = oldData if isinstance(oldData, _SplineUserData) else _SplineUserData()
        data.points = []
        data.frame_lines = []

        data.draw_spline = _read_bool(node, TwistSplineNode.aSplineDisplay, True)
        debug = _read_bool(node, TwistSplineNode.aDebugDisplay, False)
        scale = _read_double(node, TwistSplineNode.aDebugScale, 1.0)

        spline = _read_spline(node)
        if spline is None or not spline.segments:
            return data

        for p in spline.get_points():
            data.points.append(om.MPoint(p[0], p[1], p[2]))

        if debug:
            red = om.MColor((1.0, 0.1, 0.1))
            green = om.MColor((0.1, 1.0, 0.1))
            blue = om.MColor((0.2, 0.3, 1.0))
            lo, hi = spline.param_range
            samples = 24
            for i in range(samples):
                t = lo + (hi - lo) * i / (samples - 1)
                f = spline.matrix_at_param(t)
                o = om.MPoint(f.tran[0], f.tran[1], f.tran[2])
                for axis, col in ((f.tan, red), (f.norm, green), (f.binorm, blue)):
                    tip = om.MPoint(o[0] + axis[0] * scale,
                                    o[1] + axis[1] * scale,
                                    o[2] + axis[2] * scale)
                    data.frame_lines.append((o, tip, col))
        return data

    def addUIDrawables(self, objPath, drawManager, frameContext, data):
        if not isinstance(data, _SplineUserData):
            return

        drawManager.beginDrawable()
        if data.draw_spline and len(data.points) > 1:
            drawManager.setColor(om.MColor((0.9, 0.9, 0.2)))
            for i in range(len(data.points) - 1):
                drawManager.line(data.points[i], data.points[i + 1])

        for a, b, col in data.frame_lines:
            drawManager.setColor(col)
            drawManager.line(a, b)
        drawManager.endDrawable()
