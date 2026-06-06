"""
``pyTwistTangent`` -- Python/OM2 port of the C++ ``twistTangent`` node.

Faithful transcription of ``twistTangentNode.cpp::compute`` (three branches),
with the vector math living in :mod:`twistspline.tangent`. Matrix translation /
Y-axis extraction and ``parentInverseMatrix`` application use OM2 directly so
they match the C++ exactly. Distinct type name + MTypeId to coexist with the
C++ plugin.
"""

import maya.api.OpenMaya as om

from ..tangent import smooth_tangent, linear_target, out_tangent


def maya_useNewAPI():
    pass


def _trans(m):
    t = om.MTransformationMatrix(m).translation(om.MSpace.kWorld)
    return [t.x, t.y, t.z]


def _row(m, r):
    return [m.getElement(r, 0), m.getElement(r, 1), m.getElement(r, 2)]


class TwistTangentNode(om.MPxNode):
    kId = om.MTypeId(0x0013F743)
    kName = "pyTwistTangent"

    aOut = aOutX = aOutY = aOutZ = None
    aSmoothTan = aSmoothTanX = aSmoothTanY = aSmoothTanZ = None
    aOutLinearTarget = aOutLinearTargetX = aOutLinearTargetY = aOutLinearTargetZ = None
    aOutTwistUp = aOutTwistUpX = aOutTwistUpY = aOutTwistUpZ = None
    aOutTwistMat = None
    aParentInverseMatrix = aInTangent = None
    aPrevVertex = aCurrentVertex = aNextVertex = None
    aNextLinearTarget = aNextLinearTargetX = aNextLinearTargetY = aNextLinearTargetZ = None
    aAuto = aSmooth = aWeight = aBackpoint = aEndpoint = None

    def __init__(self):
        super(TwistTangentNode, self).__init__()

    @staticmethod
    def creator():
        return TwistTangentNode()

    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        uAttr = om.MFnUnitAttribute()
        mAttr = om.MFnMatrixAttribute()
        cls = TwistTangentNode
        add = om.MPxNode.addAttribute
        kDist = om.MFnUnitAttribute.kDistance

        def _vec_out(long, short, px):
            x = uAttr.create(long + "X", short + "x", kDist, 0.0)
            uAttr.writable = False
            y = uAttr.create(long + "Y", short + "y", kDist, 0.0)
            uAttr.writable = False
            z = uAttr.create(long + "Z", short + "z", kDist, 0.0)
            uAttr.writable = False
            p = nAttr.create(long, px, x, y, z)
            add(p)
            return p, x, y, z

        cls.aOut, cls.aOutX, cls.aOutY, cls.aOutZ = _vec_out("out", "o", "out")
        cls.aSmoothTan, cls.aSmoothTanX, cls.aSmoothTanY, cls.aSmoothTanZ = \
            _vec_out("smoothTan", "st", "st")
        cls.aOutLinearTarget, cls.aOutLinearTargetX, cls.aOutLinearTargetY, cls.aOutLinearTargetZ = \
            _vec_out("outLinearTarget", "lt", "lt")
        cls.aOutTwistUp, cls.aOutTwistUpX, cls.aOutTwistUpY, cls.aOutTwistUpZ = \
            _vec_out("outTwistUp", "ot", "ot")

        cls.aOutTwistMat = mAttr.create("outTwistMat", "otm")
        mAttr.writable = False
        add(cls.aOutTwistMat)

        def _mat_in(long, short, hidden=True):
            a = mAttr.create(long, short)
            mAttr.hidden = hidden
            add(a)
            return a

        cls.aParentInverseMatrix = _mat_in("parentInverseMatrix", "pim")
        cls.aInTangent = _mat_in("inTangent", "it")
        cls.aPrevVertex = _mat_in("previousVertex", "pv")
        cls.aCurrentVertex = _mat_in("currentVertex", "cv")
        cls.aNextVertex = _mat_in("nextVertex", "nv")

        nltx = uAttr.create("inLinearTargetX", "nltx", kDist, 0.0)
        nlty = uAttr.create("inLinearTargetY", "nlty", kDist, 0.0)
        nltz = uAttr.create("inLinearTargetZ", "nltz", kDist, 0.0)
        cls.aNextLinearTarget = nAttr.create("inLinearTarget", "nlt", nltx, nlty, nltz)
        cls.aNextLinearTargetX, cls.aNextLinearTargetY, cls.aNextLinearTargetZ = nltx, nlty, nltz
        add(cls.aNextLinearTarget)

        cls.aAuto = nAttr.create("auto", "a", om.MFnNumericData.kDouble, 0.0)
        nAttr.setMin(0.0); nAttr.setMax(1.0); nAttr.keyable = True
        add(cls.aAuto)
        cls.aSmooth = nAttr.create("smooth", "s", om.MFnNumericData.kDouble, 1.0)
        nAttr.setMin(0.0); nAttr.setMax(1.0); nAttr.keyable = True
        add(cls.aSmooth)
        cls.aWeight = nAttr.create("weight", "w", om.MFnNumericData.kDouble, 1.0)
        nAttr.setMin(0.0); nAttr.setMax(3.0); nAttr.keyable = True
        add(cls.aWeight)
        cls.aBackpoint = nAttr.create("backpoint", "bp", om.MFnNumericData.kBoolean, False)
        add(cls.aBackpoint)
        cls.aEndpoint = nAttr.create("endpoint", "ep", om.MFnNumericData.kBoolean, False)
        add(cls.aEndpoint)

        aa = om.MPxNode.attributeAffects
        smooth_src = (cls.aPrevVertex, cls.aCurrentVertex, cls.aNextVertex,
                      cls.aWeight, cls.aBackpoint, cls.aEndpoint)
        smooth_dst = (cls.aSmoothTan, cls.aSmoothTanX, cls.aSmoothTanY, cls.aSmoothTanZ,
                      cls.aOutTwistUp, cls.aOutTwistUpX, cls.aOutTwistUpY, cls.aOutTwistUpZ,
                      cls.aOutTwistMat)
        for s in smooth_src:
            for t in smooth_dst:
                aa(s, t)

        lin_src = (cls.aSmooth, cls.aSmoothTan, cls.aCurrentVertex, cls.aWeight,
                   cls.aPrevVertex, cls.aNextVertex)
        lin_dst = (cls.aOutLinearTarget, cls.aOutLinearTargetX,
                   cls.aOutLinearTargetY, cls.aOutLinearTargetZ)
        for s in lin_src:
            for t in lin_dst:
                aa(s, t)

        out_src = (cls.aParentInverseMatrix, cls.aInTangent, cls.aCurrentVertex,
                   cls.aSmoothTan, cls.aNextLinearTarget, cls.aSmooth, cls.aAuto,
                   cls.aWeight, cls.aPrevVertex, cls.aNextVertex)
        out_dst = (cls.aOut, cls.aOutX, cls.aOutY, cls.aOutZ)
        for s in out_src:
            for t in out_dst:
                aa(s, t)

    # -----------------------------------------------------------------
    def compute(self, plug, data):
        cls = TwistTangentNode
        attr = plug.attribute()

        smooth_grp = (cls.aSmoothTan, cls.aSmoothTanX, cls.aSmoothTanY, cls.aSmoothTanZ,
                      cls.aOutTwistUp, cls.aOutTwistUpX, cls.aOutTwistUpY, cls.aOutTwistUpZ,
                      cls.aOutTwistMat)
        lin_grp = (cls.aOutLinearTarget, cls.aOutLinearTargetX,
                   cls.aOutLinearTargetY, cls.aOutLinearTargetZ)
        out_grp = (cls.aOut, cls.aOutX, cls.aOutY, cls.aOutZ)

        if attr in smooth_grp:
            cur_m = data.inputValue(cls.aCurrentVertex).asMatrix()
            pre = _trans(data.inputValue(cls.aPrevVertex).asMatrix())
            cur = _trans(cur_m)
            nxt = _trans(data.inputValue(cls.aNextVertex).asMatrix())
            cur_y = _row(cur_m, 1)
            weight = data.inputValue(cls.aWeight).asDouble()
            bp = data.inputValue(cls.aBackpoint).asBool()
            ep = data.inputValue(cls.aEndpoint).asBool()

            smo, nrm, tan, binv = smooth_tangent(pre, cur, nxt, cur_y, weight, bp, ep)

            data.outputValue(cls.aOutTwistUp).setMVector(om.MVector(nrm))
            data.outputValue(cls.aOutTwistUp).setClean()
            data.outputValue(cls.aSmoothTan).setMVector(om.MVector(smo))
            data.outputValue(cls.aSmoothTan).setClean()

            inv_par = data.inputValue(cls.aParentInverseMatrix).asMatrix()
            out_mat = om.MMatrix([
                tan[0], tan[1], tan[2], 0.0,
                nrm[0], nrm[1], nrm[2], 0.0,
                binv[0], binv[1], binv[2], 0.0,
                cur[0], cur[1], cur[2], 1.0,
            ])
            mh = data.outputValue(cls.aOutTwistMat)
            mh.setMMatrix(out_mat * inv_par)
            mh.setClean()

        elif attr in lin_grp:
            smo = data.inputValue(cls.aSmoothTan).asVector()
            smooth = data.inputValue(cls.aSmooth).asDouble()
            cur = _trans(data.inputValue(cls.aCurrentVertex).asMatrix())
            lt = linear_target([smo.x, smo.y, smo.z], smooth, cur)
            h = data.outputValue(cls.aOutLinearTarget)
            h.setMVector(om.MVector(lt))
            h.setClean()

        elif attr in out_grp:
            in_pos = _trans(data.inputValue(cls.aInTangent).asMatrix())
            inv_par = data.inputValue(cls.aParentInverseMatrix).asMatrix()
            cur = _trans(data.inputValue(cls.aCurrentVertex).asMatrix())
            smo = data.inputValue(cls.aSmoothTan).asVector()
            nlt = data.inputValue(cls.aNextLinearTarget).asVector()
            smooth = data.inputValue(cls.aSmooth).asDouble()
            auto = data.inputValue(cls.aAuto).asDouble()

            result = out_tangent(in_pos, cur, [smo.x, smo.y, smo.z],
                                 [nlt.x, nlt.y, nlt.z], smooth, auto)
            out = om.MPoint(result[0], result[1], result[2]) * inv_par
            h = data.outputValue(cls.aOut)
            h.set3Double(out[0], out[1], out[2])
            h.setClean()
        else:
            return None
        return self
