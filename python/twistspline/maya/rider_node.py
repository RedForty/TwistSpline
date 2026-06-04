"""
``pyRiderConstraint`` -- the Python/OM2 port of the C++ ``riderConstraint``.

It consumes one or more ``pyTwistSplineData`` splines plus an array of params,
runs the exact C++ param pipeline (in :func:`twistspline.rider.solve_rider`),
then builds the oriented output transforms using Maya's own
``MMatrix``/``MQuaternion``/``MEulerRotation`` -- so quaternion construction,
multi-spline slerp, ``parentInverseMatrix`` application and euler decomposition
are bit-identical to ``riderConstraint.cpp`` (lines 587-670).

Distinct node-type name + MTypeId so it loads beside the C++ plugin.
"""

import math

import maya.api.OpenMaya as om

from ..rider import solve_rider
from .spline_data import TwistSplineData


def maya_useNewAPI():
    pass


def _quat_from_frame(frame):
    """Build the C++ output rotation (rows tan/norm/binorm) and return its quat."""
    t, n, b = frame.tan, frame.norm, frame.binorm
    m = om.MMatrix([
        t[0], t[1], t[2], 0.0,
        n[0], n[1], n[2], 0.0,
        b[0], b[1], b[2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
    return om.MTransformationMatrix(m).rotation(asQuaternion=True)


def _slerp(a, b, t):
    """Shortest-path slerp of two MQuaternions (multi-spline weighting path)."""
    cs = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    if cs < 0.0:
        b = om.MQuaternion(-b.x, -b.y, -b.z, -b.w)
        cs = -cs
    if cs > 0.9995:
        r = om.MQuaternion(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                           a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t)
        return r.normal()
    theta0 = math.acos(max(-1.0, min(1.0, cs)))
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return om.MQuaternion(a.x * s0 + b.x * s1, a.y * s0 + b.y * s1,
                          a.z * s0 + b.z * s1, a.w * s0 + b.w * s1)


class RiderConstraint(om.MPxNode):
    kId = om.MTypeId(0x0013F742)
    kName = "pyRiderConstraint"

    # single inputs
    aRotateOrder = None
    aGlobalOffset = None
    aGlobalSpread = None
    aScaleCompensation = None
    aUseCycle = None
    aNormalize = None
    aNormValue = None
    aUseGlobalMin = None
    aMinGlobalParam = None
    aUseGlobalMax = None
    aMaxGlobalParam = None
    # input spline array
    aInputSplines = None
    aSpline = None
    aSplineLength = None
    aEndParam = None
    aWeight = None
    # input param array
    aParams = None
    aParam = None
    aUseMin = None
    aMinParam = None
    aUseMax = None
    aMaxParam = None
    aParentInverseMatrix = None
    # outputs
    aOutputs = None
    aTranslate = aTranslateX = aTranslateY = aTranslateZ = None
    aRotate = aRotateX = aRotateY = aRotateZ = None
    aScale = aScaleX = aScaleY = aScaleZ = None

    def __init__(self):
        super(RiderConstraint, self).__init__()

    @staticmethod
    def creator():
        return RiderConstraint()

    # -----------------------------------------------------------------
    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        uAttr = om.MFnUnitAttribute()
        mAttr = om.MFnMatrixAttribute()
        tAttr = om.MFnTypedAttribute()
        eAttr = om.MFnEnumAttribute()
        cAttr = om.MFnCompoundAttribute()
        cls = RiderConstraint
        add = om.MPxNode.addAttribute

        cls.aRotateOrder = eAttr.create("rotateOrder", "ro", 0)
        eAttr.keyable = True
        for nm, v in (("xyz", 0), ("yzx", 1), ("zxy", 2), ("xzy", 3), ("yxz", 4), ("zyx", 5)):
            eAttr.addField(nm, v)
        add(cls.aRotateOrder)

        cls.aGlobalOffset = nAttr.create("globalOffset", "go", om.MFnNumericData.kDouble, 0.0)
        nAttr.keyable = True
        add(cls.aGlobalOffset)
        cls.aGlobalSpread = nAttr.create("globalSpread", "gs", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        add(cls.aGlobalSpread)
        cls.aScaleCompensation = nAttr.create("scaleCompensation", "sclcmp", om.MFnNumericData.kDouble, 1.0)
        add(cls.aScaleCompensation)
        cls.aUseCycle = nAttr.create("useCycle", "uc", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        add(cls.aUseCycle)
        cls.aNormalize = nAttr.create("normalize", "n", om.MFnNumericData.kDouble, 1.0)
        nAttr.setMin(0.0)
        nAttr.setMax(1.0)
        nAttr.keyable = True
        add(cls.aNormalize)
        cls.aNormValue = nAttr.create("normValue", "nv", om.MFnNumericData.kDouble, 1.0)
        nAttr.setMin(0.0)
        nAttr.keyable = True
        add(cls.aNormValue)
        cls.aUseGlobalMin = nAttr.create("useGlobalMin", "ugn", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        add(cls.aUseGlobalMin)
        cls.aMinGlobalParam = nAttr.create("minGlobalParam", "ngp", om.MFnNumericData.kDouble, 0.0)
        nAttr.keyable = True
        add(cls.aMinGlobalParam)
        cls.aUseGlobalMax = nAttr.create("useGlobalMax", "ugx", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        add(cls.aUseGlobalMax)
        cls.aMaxGlobalParam = nAttr.create("maxGlobalParam", "xgp", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        add(cls.aMaxGlobalParam)

        # spline input array
        cls.aSpline = tAttr.create("spline", "s", TwistSplineData.kId)
        tAttr.hidden = True
        cls.aSplineLength = uAttr.create("splineLength", "sl", om.MFnUnitAttribute.kDistance, 0.0)
        uAttr.keyable = True
        cls.aEndParam = nAttr.create("endParam", "ep", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        cls.aWeight = nAttr.create("weight", "w", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        cls.aInputSplines = cAttr.create("inputSplines", "is")
        cAttr.array = True
        cAttr.addChild(cls.aSpline)
        cAttr.addChild(cls.aSplineLength)
        cAttr.addChild(cls.aEndParam)
        cAttr.addChild(cls.aWeight)
        add(cls.aInputSplines)

        # param input array
        cls.aParam = nAttr.create("param", "p", om.MFnNumericData.kDouble, 0.0)
        nAttr.keyable = True
        cls.aUseMin = nAttr.create("useMin", "un", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        cls.aMinParam = nAttr.create("minParam", "np", om.MFnNumericData.kDouble, 0.0)
        nAttr.keyable = True
        cls.aUseMax = nAttr.create("useMax", "ux", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        cls.aMaxParam = nAttr.create("maxParam", "xp", om.MFnNumericData.kDouble, 1.0)
        nAttr.keyable = True
        cls.aParentInverseMatrix = mAttr.create("parentInverseMatrix", "pim", om.MFnMatrixAttribute.kDouble)
        mAttr.hidden = True
        cls.aParams = cAttr.create("params", "ps")
        cAttr.array = True
        cAttr.addChild(cls.aParam)
        cAttr.addChild(cls.aUseMin)
        cAttr.addChild(cls.aMinParam)
        cAttr.addChild(cls.aUseMax)
        cAttr.addChild(cls.aMaxParam)
        cAttr.addChild(cls.aParentInverseMatrix)
        add(cls.aParams)

        # outputs
        cls.aTranslateX = uAttr.create("translateX", "tx", om.MFnUnitAttribute.kDistance, 0.0)
        uAttr.writable = False; uAttr.storable = False
        cls.aTranslateY = uAttr.create("translateY", "ty", om.MFnUnitAttribute.kDistance, 0.0)
        uAttr.writable = False; uAttr.storable = False
        cls.aTranslateZ = uAttr.create("translateZ", "tz", om.MFnUnitAttribute.kDistance, 0.0)
        uAttr.writable = False; uAttr.storable = False
        cls.aTranslate = nAttr.create("translate", "t", cls.aTranslateX, cls.aTranslateY, cls.aTranslateZ)
        nAttr.hidden = True; nAttr.writable = False; nAttr.storable = False

        cls.aRotateX = uAttr.create("rotateX", "rotx", om.MFnUnitAttribute.kAngle, 0.0)
        uAttr.writable = False; uAttr.storable = False
        cls.aRotateY = uAttr.create("rotateY", "roty", om.MFnUnitAttribute.kAngle, 0.0)
        uAttr.writable = False; uAttr.storable = False
        cls.aRotateZ = uAttr.create("rotateZ", "rotz", om.MFnUnitAttribute.kAngle, 0.0)
        uAttr.writable = False; uAttr.storable = False
        cls.aRotate = nAttr.create("rotate", "rot", cls.aRotateX, cls.aRotateY, cls.aRotateZ)
        nAttr.hidden = True; nAttr.writable = False

        cls.aScaleX = nAttr.create("scaleX", "sclx", om.MFnNumericData.kDouble, 0.0)
        nAttr.writable = False; nAttr.storable = False
        cls.aScaleY = nAttr.create("scaleY", "scly", om.MFnNumericData.kDouble, 0.0)
        nAttr.writable = False; nAttr.storable = False
        cls.aScaleZ = nAttr.create("scaleZ", "sclz", om.MFnNumericData.kDouble, 0.0)
        nAttr.writable = False; nAttr.storable = False
        cls.aScale = nAttr.create("scale", "scl", cls.aScaleX, cls.aScaleY, cls.aScaleZ)
        nAttr.hidden = True; nAttr.writable = False; nAttr.storable = False

        cls.aOutputs = cAttr.create("outputs", "out")
        cAttr.hidden = True
        cAttr.array = True
        cAttr.usesArrayDataBuilder = True
        cAttr.addChild(cls.aTranslate)
        cAttr.addChild(cls.aRotate)
        cAttr.addChild(cls.aScale)
        add(cls.aOutputs)

        # affects: every input affects every output (mirrors the C++ double loop)
        inputs = (cls.aRotateOrder, cls.aInputSplines, cls.aSpline, cls.aSplineLength,
                  cls.aEndParam, cls.aWeight, cls.aParams, cls.aParam, cls.aParentInverseMatrix,
                  cls.aGlobalOffset, cls.aUseCycle, cls.aGlobalSpread, cls.aNormalize,
                  cls.aNormValue, cls.aUseMin, cls.aMinParam, cls.aUseMax, cls.aMaxParam,
                  cls.aUseGlobalMin, cls.aMinGlobalParam, cls.aUseGlobalMax,
                  cls.aMaxGlobalParam, cls.aScaleCompensation)
        outputs = (cls.aOutputs, cls.aTranslate, cls.aRotate, cls.aScale,
                   cls.aTranslateX, cls.aRotateX, cls.aScaleX,
                   cls.aTranslateY, cls.aRotateY, cls.aScaleY,
                   cls.aTranslateZ, cls.aRotateZ, cls.aScaleZ)
        for ii in inputs:
            for oo in outputs:
                om.MPxNode.attributeAffects(ii, oo)

    # -----------------------------------------------------------------
    def compute(self, plug, data):
        cls = RiderConstraint
        outish = (cls.aOutputs, cls.aTranslate, cls.aTranslateX, cls.aTranslateY,
                  cls.aTranslateZ, cls.aRotate, cls.aRotateX, cls.aRotateY, cls.aRotateZ,
                  cls.aScale, cls.aScaleX, cls.aScaleY, cls.aScaleZ)
        if plug not in outish:
            return None

        # ---- gather splines (weight > 0, valid data) ----
        splines, weights, spline_lens, end_params = [], [], [], []
        spAH = data.inputArrayValue(cls.aInputSplines)
        for i in range(len(spAH)):
            spAH.jumpToPhysicalElement(i)
            grp = spAH.inputValue()
            w = grp.child(cls.aWeight).asDouble()
            if w <= 0.0:
                continue
            obj = grp.child(cls.aSpline).data()
            if obj.isNull():
                continue
            try:
                spline = om.MFnPluginData(obj).data().spline()
            except Exception:
                continue
            if spline is None or not spline.segments:
                continue
            splines.append(spline)
            weights.append(w)
            spline_lens.append(grp.child(cls.aSplineLength).asDouble())
            end_params.append(grp.child(cls.aEndParam).asDouble())

        outArray = data.outputArrayValue(cls.aOutputs)
        if not splines:
            outArray.setAllClean()
            return self

        # ---- gather params (sparse logical indices, like the C++) ----
        pAH = data.inputArrayValue(cls.aParams)
        by_index = {}
        max_idx = -1
        for i in range(len(pAH)):
            pAH.jumpToPhysicalElement(i)
            li = pAH.elementIndex()
            grp = pAH.inputValue()
            by_index[li] = (
                grp.child(cls.aParam).asDouble(),
                grp.child(cls.aUseMin).asBool(),
                grp.child(cls.aMinParam).asDouble(),
                grp.child(cls.aUseMax).asBool(),
                grp.child(cls.aMaxParam).asDouble(),
                om.MMatrix(grp.child(cls.aParentInverseMatrix).asMatrix()),
            )
            max_idx = max(max_idx, li)

        param_inputs, inv_pars = [], []
        for idx in range(max_idx + 1):
            if idx in by_index:
                p, umin, pmin, umax, pmax, ipm = by_index[idx]
            else:
                p, umin, pmin, umax, pmax, ipm = 0.0, False, 0.0, False, 1.0, om.MMatrix()
            param_inputs.append((p, umin, pmin, umax, pmax))
            inv_pars.append(ipm)

        # ---- globals ----
        g_off = data.inputValue(cls.aGlobalOffset).asDouble()
        g_spread = data.inputValue(cls.aGlobalSpread).asDouble()
        sclcmp = data.inputValue(cls.aScaleCompensation).asDouble()
        use_cycle = data.inputValue(cls.aUseCycle).asBool()
        do_norm = data.inputValue(cls.aNormalize).asDouble()
        norm_val = data.inputValue(cls.aNormValue).asDouble()
        use_gmin = data.inputValue(cls.aUseGlobalMin).asBool()
        gmin = data.inputValue(cls.aMinGlobalParam).asDouble()
        use_gmax = data.inputValue(cls.aUseGlobalMax).asBool()
        gmax = data.inputValue(cls.aMaxGlobalParam).asDouble()
        order = data.inputValue(cls.aRotateOrder).asShort()

        weights, frames, twisted = solve_rider(
            splines, weights, spline_lens, end_params, param_inputs,
            global_offset=g_off, global_spread=g_spread, scale_compensation=sclcmp,
            use_cycle=use_cycle, normalize=do_norm, norm_value=norm_val,
            use_global_min=use_gmin, min_global_param=gmin,
            use_global_max=use_gmax, max_global_param=gmax)

        nsplines = len(splines)

        builder = outArray.builder()
        for pIdx in range(len(param_inputs)):
            if twisted:
                fr = frames[0][pIdx]
                tran = [fr.tran[0], fr.tran[1], fr.tran[2]]
                scl = [fr.scl[0], fr.scl[1], fr.scl[2]]
                quat = _quat_from_frame(fr)
            else:
                tran = [0.0, 0.0, 0.0]
                scl = [0.0, 0.0, 0.0]
                tw = 0.0
                for sIdx in range(nsplines):
                    wv = weights[sIdx]
                    fr = frames[sIdx][pIdx]
                    tran = [tran[k] + fr.tran[k] * wv for k in range(3)]
                    scl = [scl[k] + fr.scl[k] * wv for k in range(3)]
                    tw += fr.twist * wv
                prev = _quat_from_frame(frames[0][pIdx]).normal()
                for sIdx in range(1, nsplines):
                    cur = _quat_from_frame(frames[sIdx][pIdx])
                    pw = weights[sIdx - 1]
                    cw = weights[sIdx]
                    prev = _slerp(prev, cur, cw / (pw + cw))
                xt = om.MQuaternion(tw, om.MVector(1.0, 0.0, 0.0))
                quat = xt * prev

            inv_par = inv_pars[pIdx]
            tranP = om.MPoint(tran[0], tran[1], tran[2]) * inv_par
            qmat = quat.asMatrix() * inv_par
            eul = om.MEulerRotation.decompose(qmat, order)

            outH = builder.addElement(pIdx)
            outH.child(cls.aTranslate).set3Double(tranP[0], tranP[1], tranP[2])
            outH.child(cls.aRotate).set3Double(eul.x, eul.y, eul.z)
            outH.child(cls.aScale).set3Double(scl[0], scl[1], scl[2])

        outArray.set(builder)
        outArray.setAllClean()
        return self
