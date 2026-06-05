"""
``pyTwistMultiTangent`` -- Python/OM2 port of the C++ ``twistMultiTangent`` node.

This node computes every in/out bezier tangent for the whole control hull at
once. It is the most matrix/vector-heavy of the four nodes, so it is transcribed
directly against OM2's ``MVector``/``MMatrix`` (whose ``^`` cross, ``*`` dot,
``.normal()``, ``.inverse()`` and matrix-multiply map 1:1 onto the C++) rather
than split into a Maya-free kernel. Correctness is established by the in-Maya
parity test against the C++ node (see build.create_multi_tangent_parity).

Faithful to twistMultiTangentNode.cpp:356-681, including the open/closed
endpoint handling and the (startTension-at-both-ends) behavior of the C++.
"""

import copy

import maya.api.OpenMaya as om


def maya_useNewAPI():
    pass


def _v(m, r):
    """Row r of MMatrix as an MVector (x,y,z)."""
    return om.MVector(m.getElement(r, 0), m.getElement(r, 1), m.getElement(r, 2))


def _build_mat(tfm, inrm, tan):
    """Graham-Schmidt orthonormal matrix with rows tan, nrm, bin and trans tfm."""
    binv = (tan ^ inrm).normal()
    nrm = (binv ^ tan).normal()
    return om.MMatrix([
        tan[0], tan[1], tan[2], 0.0,
        nrm[0], nrm[1], nrm[2], 0.0,
        binv[0], binv[1], binv[2], 0.0,
        tfm[0], tfm[1], tfm[2], 1.0,
    ])


def _build_vert_tangent(next_len, pre_len, next_nrm, pre_nrm):
    """Normalized curve tangent at a vertex (twistMultiTangentNode.cpp:406-444)."""
    if pre_len == 0.0:
        if next_len == 0.0:
            return om.MVector(0.0, 1.0, 0.0)
        return om.MVector(next_nrm)
    if next_len == 0.0:
        return -pre_nrm
    dot = pre_nrm * next_nrm
    if dot > 0.999999999:
        y = om.MVector(0.0, 0.0, 0.0)
        if abs(pre_nrm * y) > 0.999999999:
            y = om.MVector(1.0, 0.0, 0.0)
        else:
            y = om.MVector(0.0, 1.0, 0.0)
        return (pre_nrm ^ y).normal()
    elif dot < -0.999999999:
        return -pre_nrm
    binv = (pre_nrm ^ next_nrm).normal()
    return ((binv ^ pre_nrm) + (binv ^ next_nrm)).normal()


class _Dir(object):
    __slots__ = ("weight", "smooth", "autoVal", "piMat", "userMat",
                 "leg", "legLen", "normLeg", "smoothTan", "linearTan", "doneTan")

    def __init__(self):
        self.weight = 1.0
        self.smooth = 1.0
        self.autoVal = 1.0
        self.piMat = om.MMatrix()
        self.userMat = om.MMatrix()
        self.leg = om.MVector()
        self.legLen = 0.0
        self.normLeg = om.MVector()
        self.smoothTan = om.MVector()
        self.linearTan = om.MVector()
        self.doneTan = om.MVector()


class _Tan(object):
    __slots__ = ("tfmMat", "smoothMat", "tfm", "norm", "piTwist", "inTan", "outTan")

    def __init__(self):
        self.tfmMat = om.MMatrix()
        self.smoothMat = om.MMatrix()
        self.tfm = om.MVector()
        self.norm = om.MVector()
        self.piTwist = om.MMatrix()
        self.inTan = _Dir()
        self.outTan = _Dir()


def _build_smooth_mats(dat, start_tension, end_tension, closed):
    n = len(dat)
    for i in range(1, n - 1):
        cur = dat[i]
        tan = _build_vert_tangent(cur.outTan.legLen, cur.inTan.legLen,
                                  cur.outTan.normLeg, cur.inTan.normLeg)
        cur.smoothMat = _build_mat(cur.tfm, cur.norm, tan)
        pre_len = cur.inTan.legLen * cur.inTan.weight / 3.0
        next_len = cur.outTan.legLen * cur.outTan.weight / 3.0
        cur.inTan.smoothTan = -tan * pre_len
        cur.outTan.smoothTan = tan * next_len

    if closed:
        start = dat[0]
        end = dat[n - 1]
        tan = _build_vert_tangent(start.outTan.legLen, end.inTan.legLen,
                                  start.outTan.normLeg, end.inTan.normLeg)
        start.smoothMat = _build_mat(start.tfm, start.norm, tan)
        end.smoothMat = start.smoothMat
        pre_len = end.inTan.legLen * end.inTan.weight / 3.0
        next_len = start.outTan.legLen * start.outTan.weight / 3.0
        end.inTan.smoothTan = -tan * pre_len
        start.outTan.smoothTan = tan * next_len
    else:
        dat[0].outTan.smoothTan = (
            (dat[1].tfm + dat[1].inTan.smoothTan * start_tension - dat[0].tfm)
            * (dat[0].outTan.weight / 2.0))
        dat[0].smoothMat = _build_mat(dat[0].tfm, dat[0].norm,
                                      dat[0].outTan.smoothTan.normal())
        s = n
        dat[s - 1].inTan.smoothTan = (
            (dat[s - 2].tfm + dat[s - 2].outTan.smoothTan * start_tension - dat[s - 1].tfm)
            * (dat[s - 1].inTan.weight / 2.0))
        dat[s - 1].smoothMat = _build_mat(dat[s - 1].tfm, dat[s - 1].norm,
                                          (-dat[s - 1].inTan.smoothTan).normal())


def _build_linear_tangents(dat):
    n = len(dat)
    for i in range(1, n - 1):
        prev, cur, nxt = dat[i - 1], dat[i], dat[i + 1]
        pre_len = cur.inTan.legLen * cur.inTan.weight / 3.0
        next_len = cur.outTan.legLen * cur.outTan.weight / 3.0
        in_dir = (prev.tfm + (prev.outTan.smoothTan * prev.outTan.smooth) - cur.tfm).normal()
        out_dir = (nxt.tfm + (nxt.inTan.smoothTan * nxt.inTan.smooth) - cur.tfm).normal()
        cur.inTan.linearTan = in_dir * pre_len
        cur.outTan.linearTan = out_dir * next_len

    s = n
    dat[0].outTan.linearTan = (
        (dat[1].tfm + (dat[1].inTan.smoothTan * dat[1].inTan.smooth) - dat[0].tfm)
        / (3.0 - dat[1].inTan.smooth))
    dat[s - 1].inTan.linearTan = (
        (dat[s - 2].tfm + (dat[s - 2].outTan.smoothTan * dat[s - 2].outTan.smooth) - dat[s - 1].tfm)
        / (3.0 - dat[s - 2].outTan.smooth))


def _build_done_tangents(dat):
    for cur in dat:
        ti = cur.tfmMat.inverse()

        in_smo = cur.inTan.smooth
        in_auto = cur.inTan.autoVal
        in_user_mat = ti * cur.inTan.userMat
        in_user_tan = _v(in_user_mat, 3)
        in_auto_tan = cur.inTan.smoothTan * in_smo + cur.inTan.linearTan * (1.0 - in_smo)
        cur.inTan.doneTan = in_auto_tan * in_auto + in_user_tan * (1.0 - in_auto)

        out_smo = cur.outTan.smooth
        out_auto = cur.outTan.autoVal
        out_user_mat = ti * cur.outTan.userMat
        out_user_tan = _v(out_user_mat, 3)
        out_auto_tan = cur.outTan.smoothTan * out_smo + cur.outTan.linearTan * (1.0 - out_smo)
        cur.outTan.doneTan = out_auto_tan * out_auto + out_user_tan * (1.0 - out_auto)


class TwistMultiTangentNode(om.MPxNode):
    kId = om.MTypeId(0x0013F744)
    kName = "pyTwistMultiTangent"

    # inputs
    aVertMat = aInParentInv = aOutParentInv = aTwistParentInv = None
    aInTanWeight = aOutTanWeight = aInSmooth = aOutSmooth = None
    aInTanAuto = aOutTanAuto = aInTanMat = aOutTanMat = None
    aVertData = None
    aStartTension = aEndTension = aMaxVertices = aClosed = None
    # outputs
    aInTan = aInTanX = aInTanY = aInTanZ = None
    aOutTan = aOutTanX = aOutTanY = aOutTanZ = None
    aInTanLen = aOutTanLen = None
    aOutTwistUp = aOutTwistUpX = aOutTwistUpY = aOutTwistUpZ = None
    aOutTwistMat = None
    aTangents = None

    def __init__(self):
        super(TwistMultiTangentNode, self).__init__()

    @staticmethod
    def creator():
        return TwistMultiTangentNode()

    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        uAttr = om.MFnUnitAttribute()
        mAttr = om.MFnMatrixAttribute()
        cAttr = om.MFnCompoundAttribute()
        cls = TwistMultiTangentNode
        add = om.MPxNode.addAttribute
        kDist = om.MFnUnitAttribute.kDistance

        def _mat(long, short):
            a = mAttr.create(long, short)
            mAttr.keyable = False
            mAttr.storable = False
            mAttr.hidden = True
            return a

        def _num(long, short, default, smin=None, smax=None):
            a = nAttr.create(long, short, om.MFnNumericData.kDouble, default)
            nAttr.keyable = True
            nAttr.storable = False
            if smin is not None:
                nAttr.setSoftMin(smin)
            if smax is not None:
                nAttr.setSoftMax(smax)
            return a

        cls.aInTanMat = _mat("inTanMat", "itm")
        cls.aOutTanMat = _mat("outTanMat", "otm")
        cls.aInTanWeight = _num("inTanWeight", "itw", 1.0, 0.0, 2.0)
        cls.aOutTanWeight = _num("outTanWeight", "otw", 1.0, 0.0, 2.0)
        cls.aInSmooth = _num("inSmooth", "ism", 1.0, 0.0, 1.0)
        cls.aOutSmooth = _num("outSmooth", "osm", 1.0, 0.0, 1.0)
        cls.aInTanAuto = _num("inAuto", "iat", 1.0, 0.0, 1.0)
        cls.aOutTanAuto = _num("outAuto", "oat", 1.0, 0.0, 1.0)
        cls.aVertMat = _mat("vertMat", "vm")
        cls.aInParentInv = _mat("inParentInverseMatrix", "ipim")
        cls.aOutParentInv = _mat("outParentInverseMatrix", "opim")
        cls.aTwistParentInv = _mat("twistParentInverseMatrix", "tpim")

        cls.aVertData = cAttr.create("vertData", "vd")
        cAttr.array = True
        cAttr.disconnectBehavior = om.MFnAttribute.kDelete
        for child in (cls.aVertMat, cls.aInParentInv, cls.aOutParentInv,
                      cls.aTwistParentInv, cls.aInTanWeight, cls.aOutTanWeight,
                      cls.aInSmooth, cls.aOutSmooth, cls.aInTanAuto, cls.aOutTanAuto,
                      cls.aInTanMat, cls.aOutTanMat):
            cAttr.addChild(child)
        add(cls.aVertData)

        cls.aStartTension = nAttr.create("startTension", "st", om.MFnNumericData.kDouble, 2.0)
        nAttr.keyable = True
        add(cls.aStartTension)
        cls.aEndTension = nAttr.create("endTension", "et", om.MFnNumericData.kDouble, 2.0)
        nAttr.keyable = True
        add(cls.aEndTension)
        cls.aMaxVertices = nAttr.create("maxVertices", "mv", om.MFnNumericData.kInt, 999)
        nAttr.keyable = True
        nAttr.setMin(2)
        add(cls.aMaxVertices)
        cls.aClosed = nAttr.create("closed", "cl", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        add(cls.aClosed)

        def _vec_out(long, short):
            x = uAttr.create(long + "X", short + "x", kDist, 0.0)
            uAttr.keyable = False; uAttr.storable = False; uAttr.writable = False
            y = uAttr.create(long + "Y", short + "y", kDist, 0.0)
            uAttr.keyable = False; uAttr.storable = False; uAttr.writable = False
            z = uAttr.create(long + "Z", short + "z", kDist, 0.0)
            uAttr.keyable = False; uAttr.storable = False; uAttr.writable = False
            p = nAttr.create(long, short, x, y, z)
            nAttr.keyable = False; nAttr.storable = False; nAttr.writable = False
            return p, x, y, z

        cls.aInTan, cls.aInTanX, cls.aInTanY, cls.aInTanZ = _vec_out("inVertTan", "ivt")
        # match C++ child short names ivx/ivy/ivz, ovx.. via explicit recreate
        cls.aOutTan, cls.aOutTanX, cls.aOutTanY, cls.aOutTanZ = _vec_out("outVertTan", "ovt")
        cls.aOutTwistUp, cls.aOutTwistUpX, cls.aOutTwistUpY, cls.aOutTwistUpZ = _vec_out("twistUp", "tu")

        cls.aOutTwistMat = mAttr.create("twistMat", "tm")
        mAttr.keyable = False; mAttr.writable = False; mAttr.storable = False; mAttr.hidden = True

        cls.aInTanLen = nAttr.create("inTanLen", "itl", om.MFnNumericData.kDouble, 0.0)
        nAttr.keyable = False; nAttr.writable = False; nAttr.storable = False
        cls.aOutTanLen = nAttr.create("outTanLen", "otl", om.MFnNumericData.kDouble, 0.0)
        nAttr.keyable = False; nAttr.writable = False; nAttr.storable = False

        cls.aTangents = cAttr.create("vertTans", "vt")
        for child in (cls.aInTan, cls.aOutTan, cls.aInTanLen, cls.aOutTanLen,
                      cls.aOutTwistUp, cls.aOutTwistMat):
            cAttr.addChild(child)
        cAttr.array = True
        cAttr.usesArrayDataBuilder = True
        cAttr.keyable = False
        cAttr.writable = False
        cAttr.storable = False
        add(cls.aTangents)

        ins = (cls.aStartTension, cls.aEndTension, cls.aMaxVertices, cls.aClosed,
               cls.aVertMat, cls.aInParentInv, cls.aOutParentInv, cls.aTwistParentInv,
               cls.aInTanWeight, cls.aOutTanWeight, cls.aInTanMat, cls.aOutTanMat,
               cls.aInSmooth, cls.aOutSmooth, cls.aInTanAuto, cls.aOutTanAuto, cls.aVertData)
        outs = (cls.aInTan, cls.aInTanX, cls.aInTanY, cls.aInTanZ,
                cls.aOutTan, cls.aOutTanX, cls.aOutTanY, cls.aOutTanZ,
                cls.aInTanLen, cls.aOutTanLen, cls.aOutTwistUp, cls.aOutTwistUpX,
                cls.aOutTwistUpY, cls.aOutTwistUpZ, cls.aOutTwistMat, cls.aTangents)
        for i in ins:
            for o in outs:
                om.MPxNode.attributeAffects(i, o)

    # -----------------------------------------------------------------
    def compute(self, plug, data):
        cls = TwistMultiTangentNode
        outs = (cls.aInTan, cls.aInTanX, cls.aInTanY, cls.aInTanZ,
                cls.aOutTan, cls.aOutTanX, cls.aOutTanY, cls.aOutTanZ,
                cls.aInTanLen, cls.aOutTanLen, cls.aOutTwistUp, cls.aOutTwistUpX,
                cls.aOutTwistUpY, cls.aOutTwistUpZ, cls.aOutTwistMat, cls.aTangents)
        if plug.attribute() not in outs:
            return None

        vertInputs = data.inputArrayValue(cls.aVertData)
        icount = len(vertInputs)
        start_tension = data.inputValue(cls.aStartTension).asDouble()
        end_tension = data.inputValue(cls.aEndTension).asDouble()
        raw_max = data.inputValue(cls.aMaxVertices).asInt()
        max_verts = 0 if raw_max < 0 else raw_max
        closed = data.inputValue(cls.aClosed).asBool()

        icount = min(icount, max_verts)
        out_array = data.outputArrayValue(cls.aTangents)
        if icount < 2:
            out_array.setAllClean()
            return self

        dat = []
        for i in range(icount):
            vertInputs.jumpToPhysicalElement(i)
            h = vertInputs.inputValue()
            d = _Tan()
            d.inTan.weight = h.child(cls.aInTanWeight).asDouble()
            d.outTan.weight = h.child(cls.aOutTanWeight).asDouble()
            d.inTan.smooth = h.child(cls.aInSmooth).asDouble()
            d.outTan.smooth = h.child(cls.aOutSmooth).asDouble()
            d.inTan.autoVal = h.child(cls.aInTanAuto).asDouble()
            d.outTan.autoVal = h.child(cls.aOutTanAuto).asDouble()
            vmat = h.child(cls.aVertMat).asMatrix()
            d.tfmMat = vmat
            d.tfm = _v(vmat, 3)
            d.norm = _v(vmat, 1)
            d.inTan.piMat = h.child(cls.aInParentInv).asMatrix()
            d.outTan.piMat = h.child(cls.aOutParentInv).asMatrix()
            d.piTwist = h.child(cls.aTwistParentInv).asMatrix()
            d.inTan.userMat = h.child(cls.aInTanMat).asMatrix()
            d.outTan.userMat = h.child(cls.aOutTanMat).asMatrix()
            dat.append(d)

        if closed:
            dat[0].inTan = copy.copy(dat[-1].inTan)
            dat[-1].outTan = copy.copy(dat[0].outTan)

        for i in range(1, len(dat)):
            prev, cur = dat[i - 1], dat[i]
            cur.inTan.leg = prev.tfm - cur.tfm
            cur.inTan.legLen = cur.inTan.leg.length()
            cur.inTan.normLeg = cur.inTan.leg / cur.inTan.legLen
            prev.outTan.leg = -cur.inTan.leg
            prev.outTan.legLen = cur.inTan.legLen
            prev.outTan.normLeg = -cur.inTan.normLeg

        dat[0].inTan.legLen = 0.0
        dat[-1].outTan.legLen = 0.0

        _build_smooth_mats(dat, start_tension, end_tension, closed)
        _build_linear_tangents(dat)
        _build_done_tangents(dat)

        builder = out_array.builder()
        for i in range(len(dat)):
            cur = dat[i]
            outH = builder.addElement(i)

            in_pt = om.MPoint(cur.inTan.doneTan + cur.tfm) * cur.inTan.piMat
            outH.child(cls.aInTan).set3Double(in_pt[0], in_pt[1], in_pt[2])
            outH.child(cls.aInTanLen).setDouble(cur.inTan.doneTan.length())

            out_pt = om.MPoint(cur.outTan.doneTan + cur.tfm) * cur.outTan.piMat
            outH.child(cls.aOutTan).set3Double(out_pt[0], out_pt[1], out_pt[2])
            outH.child(cls.aOutTanLen).setDouble(cur.outTan.doneTan.length())

            ori = om.MTransformationMatrix(cur.smoothMat * cur.piTwist)
            outH.child(cls.aOutTwistMat).setMMatrix(ori.rotation(asQuaternion=True).asMatrix())

            outH.child(cls.aOutTwistUp).set3Double(
                cur.tfmMat.getElement(1, 0), cur.tfmMat.getElement(1, 1),
                cur.tfmMat.getElement(1, 2))

        out_array.set(builder)
        out_array.setAllClean()
        return self
