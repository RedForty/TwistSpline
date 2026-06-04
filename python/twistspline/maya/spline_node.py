"""
``pyTwistSpline`` -- the Python/OM2 port of the C++ ``twistSpline`` locator node.

It reads the same per-CV inputs the C++ node does (interleaved control-vertex /
in-tangent / out-tangent world matrices, plus param/twist/orient values and
weights), assembles them exactly as ``twistSplineNode.cpp:308-379`` does,
builds a :class:`twistspline.core.TwistSpline` through the validated kernel, and
publishes it on the ``outputSpline`` plug as :class:`TwistSplineData`.

Node-type name and MTypeId are deliberately distinct from the C++ plugin's so
both can be loaded simultaneously (needed for parity testing).
"""

import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui

from ..core import TwistSpline
from .spline_data import TwistSplineData


def maya_useNewAPI():
    pass


def _decompose(mmatrix):
    """MMatrix -> (translation[3], quat[w,x,y,z], scale[3]) in world space."""
    tm = om.MTransformationMatrix(mmatrix)
    t = tm.translation(om.MSpace.kWorld)
    q = tm.rotation(asQuaternion=True)
    s = tm.scale(om.MSpace.kObject)
    return [t.x, t.y, t.z], [q.w, q.x, q.y, q.z], [s[0], s[1], s[2]]


class TwistSplineNode(omui.MPxLocatorNode):
    kId = om.MTypeId(0x0013F741)
    kName = "pyTwistSpline"
    kDrawClassification = "drawdb/geometry/pyTwistSpline"
    kDrawRegistrantId = "pyTwistSplinePlugin"

    # attribute handles (filled in by initialize)
    aOutputSpline = None
    aSplineLength = None
    aScaleCompensation = None
    aTwistMultiplier = None
    aMaxVertices = None
    aDebugDisplay = None
    aDebugScale = None
    aSplineDisplay = None

    aVertexData = None
    aInTangent = None
    aOutTangent = None
    aControlVertex = None
    aParamValue = None
    aParamWeight = None
    aTwistValue = None
    aTwistWeight = None
    aUseOrient = None

    def __init__(self):
        super(TwistSplineNode, self).__init__()

    @staticmethod
    def creator():
        return TwistSplineNode()

    # -----------------------------------------------------------------
    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        uAttr = om.MFnUnitAttribute()
        mAttr = om.MFnMatrixAttribute()
        tAttr = om.MFnTypedAttribute()
        cAttr = om.MFnCompoundAttribute()

        cls = TwistSplineNode

        # ---- outputs ----
        cls.aOutputSpline = tAttr.create("outputSpline", "os", TwistSplineData.kId)
        tAttr.writable = False
        tAttr.storable = False
        tAttr.hidden = True
        om.MPxNode.addAttribute(cls.aOutputSpline)

        cls.aSplineLength = nAttr.create("splineLength", "sl", om.MFnNumericData.kDouble, 0.0)
        nAttr.writable = False
        nAttr.storable = False
        om.MPxNode.addAttribute(cls.aSplineLength)

        # ---- single inputs ----
        cls.aScaleCompensation = nAttr.create("scaleCompensation", "sclcmp", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        om.MPxNode.addAttribute(cls.aScaleCompensation)

        cls.aTwistMultiplier = nAttr.create("twistMultiplier", "tm", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        om.MPxNode.addAttribute(cls.aTwistMultiplier)

        cls.aMaxVertices = nAttr.create("maxVertices", "mv", om.MFnNumericData.kInt, 999)
        om.MPxNode.addAttribute(cls.aMaxVertices)

        cls.aSplineDisplay = nAttr.create("splineDisplay", "sd", om.MFnNumericData.kBoolean, True)
        nAttr.keyable = True
        om.MPxNode.addAttribute(cls.aSplineDisplay)

        cls.aDebugDisplay = nAttr.create("debugDisplay", "dd", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        om.MPxNode.addAttribute(cls.aDebugDisplay)

        cls.aDebugScale = nAttr.create("debugScale", "ds", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        om.MPxNode.addAttribute(cls.aDebugScale)

        # ---- per-CV array compound ----
        cls.aInTangent = mAttr.create("inTangent", "int", om.MFnMatrixAttribute.kDouble)
        mAttr.hidden = True
        cls.aOutTangent = mAttr.create("outTangent", "ot", om.MFnMatrixAttribute.kDouble)
        mAttr.hidden = True
        cls.aControlVertex = mAttr.create("controlVertex", "cv", om.MFnMatrixAttribute.kDouble)
        mAttr.hidden = True

        cls.aParamValue = nAttr.create("paramValue", "pv", om.MFnNumericData.kDouble, 0.0)
        cls.aParamWeight = nAttr.create("paramWeight", "pw", om.MFnNumericData.kDouble, 0.0)
        nAttr.setMin(0.0)
        nAttr.setMax(1.0)
        cls.aTwistValue = uAttr.create("twistValue", "tv", om.MFnUnitAttribute.kAngle, 0.0)
        cls.aTwistWeight = nAttr.create("twistWeight", "tw", om.MFnNumericData.kDouble, 0.0)
        nAttr.setMin(0.0)
        nAttr.setMax(1.0)
        cls.aUseOrient = nAttr.create("useOrient", "uo", om.MFnNumericData.kDouble, 0.0)
        nAttr.setMin(0.0)
        nAttr.setMax(1.0)

        cls.aVertexData = cAttr.create("vertexData", "vd")
        cAttr.array = True
        cAttr.addChild(cls.aInTangent)
        cAttr.addChild(cls.aControlVertex)
        cAttr.addChild(cls.aOutTangent)
        cAttr.addChild(cls.aParamValue)
        cAttr.addChild(cls.aParamWeight)
        cAttr.addChild(cls.aTwistWeight)
        cAttr.addChild(cls.aUseOrient)
        cAttr.addChild(cls.aTwistValue)
        om.MPxNode.addAttribute(cls.aVertexData)

        # ---- affects ----
        for src in (cls.aVertexData, cls.aScaleCompensation, cls.aTwistMultiplier,
                    cls.aMaxVertices):
            om.MPxNode.attributeAffects(src, cls.aOutputSpline)
            om.MPxNode.attributeAffects(src, cls.aSplineLength)

    # -----------------------------------------------------------------
    def compute(self, plug, data):
        cls = TwistSplineNode
        if plug != cls.aOutputSpline and plug != cls.aSplineLength:
            return None

        scale_comp = data.inputValue(cls.aScaleCompensation).asDouble()
        twist_mul = data.inputValue(cls.aTwistMultiplier).asDouble()
        max_verts = data.inputValue(cls.aMaxVertices).asInt()

        vdArray = data.inputArrayValue(cls.aVertexData)
        ecount = len(vdArray)
        if max_verts < ecount:
            ecount = max_verts

        points, scales, quats = [], [], []
        lock_positions, lock_vals, twist_lock, user_twist, orient_lock = [], [], [], [], []
        got_locks = got_oris = False

        for i in range(ecount):
            vdArray.jumpToPhysicalElement(i)
            grp = vdArray.inputValue()

            lock_positions.append(grp.child(cls.aParamValue).asDouble() * scale_comp)
            lk = grp.child(cls.aParamWeight).asDouble()
            lock_vals.append(lk)
            got_locks = got_locks or lk > 0.0

            twist_lock.append(grp.child(cls.aTwistWeight).asDouble())
            # kAngle handle: asDouble() returns radians (matches the C++ node).
            user_twist.append(grp.child(cls.aTwistValue).asDouble() * twist_mul)

            ori = grp.child(cls.aUseOrient).asDouble()
            orient_lock.append(ori)
            got_oris = got_oris or ori > 0.0

            if i > 0:
                t, q, s = _decompose(grp.child(cls.aInTangent).asMatrix())
                points.append(t); scales.append(s); quats.append(q)

            t, q, s = _decompose(grp.child(cls.aControlVertex).asMatrix())
            points.append(t); scales.append(s); quats.append(q)

            t, q, s = _decompose(grp.child(cls.aOutTangent).asMatrix())
            points.append(t); scales.append(s); quats.append(q)

        # Drop the trailing out-tangent group (past the end).
        if quats:
            quats.pop()
        if points:
            points.pop()
        if scales:
            scales.pop()

        # Well-posing guards (twistSplineNode.cpp:375-379).
        if not got_locks and lock_vals:
            lock_vals[0] = 1.0
        if not got_oris and orient_lock:
            orient_lock[0] = 1.0

        spline = TwistSpline()
        if len(points) >= 4:
            spline.set_verts(points, scales, quats, lock_positions, lock_vals,
                             user_twist, twist_lock, orient_lock)

        # Publish the spline as plugin data.
        fnData = om.MFnPluginData()
        dataObj = fnData.create(TwistSplineData.kId)
        splineData = fnData.data()
        splineData.setSpline(spline)

        outH = data.outputValue(cls.aOutputSpline)
        outH.setMPxData(splineData)
        outH.setClean()

        lenH = data.outputValue(cls.aSplineLength)
        lenH.setDouble(spline.total_length if spline.segments else 0.0)
        lenH.setClean()

        data.setClean(plug)
        return self
