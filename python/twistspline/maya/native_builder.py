"""
Generator for the all-native-node TwistSpline rig (build-time tool).

This runs on the RIGGER's machine and emits a rig made of 100% stock Maya nodes
-- no plugin, nothing in scripts/ -- implementing the validated scheme in
``twistspline.native_ref`` node-for-node. The delivered scene has zero
dependencies; the animator just opens it.

Built in stages so each can be verified against the spec / a C++ rig:
    Stage 1 (this):   live degree-3 curve from CV controls (auto-tangents) +
                      joints riding it via motionPath + per-CV control attrs.
    Stage 2: parallel-transport RMF -> joint orientation.
    Stage 3: per-CV twist pinning.
    Stage 4: useOrient first/last (wired to the UseOrient attrs).
"""

import math

import maya.cmds as cmds

from ..core import _catmull_tangents
from ..native_ref import params_at_fractions, native_frames
from ..tangent_ref import multi_tangent_handles
from .. import _vmath as vm


# ---- tiny node-math helpers (vector ops as stock nodes) -------------------

def _decompose(ctrl):
    dm = cmds.createNode("decomposeMatrix")
    cmds.connectAttr(ctrl + ".worldMatrix[0]", dm + ".inputMatrix")
    return dm + ".outputTranslate"


def _pma(op, a, b, name):
    n = cmds.createNode("plusMinusAverage", name=name)
    cmds.setAttr(n + ".operation", op)  # 1=sum, 2=subtract
    cmds.connectAttr(a, n + ".input3D[0]")
    cmds.connectAttr(b, n + ".input3D[1]")
    return n + ".output3D"


def _add(a, b, name):
    return _pma(1, a, b, name)


def _sub(a, b, name):
    return _pma(2, a, b, name)


def _scale(a, s, name):
    n = cmds.createNode("multiplyDivide", name=name)
    cmds.connectAttr(a, n + ".input1")
    cmds.setAttr(n + ".input2", s, s, s)
    return n + ".output"


def _norm_v(a, name):
    """Normalize a 3-vector plug."""
    n = cmds.createNode("vectorProduct", name=name)
    cmds.setAttr(n + ".operation", 0)  # no-op
    cmds.setAttr(n + ".normalizeOutput", 1)
    cmds.connectAttr(a, n + ".input1")
    return n + ".output"


def _cross_v(a, b, name, normalize=False):
    """Cross product of two 3-vector plugs."""
    n = cmds.createNode("vectorProduct", name=name)
    cmds.setAttr(n + ".operation", 2)  # cross
    cmds.setAttr(n + ".normalizeOutput", 1 if normalize else 0)
    cmds.connectAttr(a, n + ".input1")
    cmds.connectAttr(b, n + ".input2")
    return n + ".output"


def _len_v(a, name):
    """Length of a 3-vector plug (scalar)."""
    n = cmds.createNode("distanceBetween", name=name)
    cmds.connectAttr(a, n + ".point1")
    cmds.setAttr(n + ".point2", 0.0, 0.0, 0.0)
    return n + ".distance"


def _add_vc(plug, const_vec, name):
    """3-vector plug + a constant vector."""
    n = cmds.createNode("plusMinusAverage", name=name)
    cmds.setAttr(n + ".operation", 1)  # sum
    cmds.connectAttr(plug, n + ".input3D[0]")
    cmds.setAttr(n + ".input3D[1]", *const_vec, type="double3")
    return n + ".output3D"


def _point_mat(vec_const, matrix_plug, name):
    """Transform a constant point by a matrix plug (incl. translation)."""
    n = cmds.createNode("vectorProduct", name=name)
    cmds.setAttr(n + ".operation", 4)  # point-matrix product
    cmds.setAttr(n + ".input1", *vec_const)
    cmds.connectAttr(matrix_plug, n + ".matrix")
    return n + ".output"


def _world_to_local(world_plug, parent_inv_plug, name):
    """A world-space point plug expressed in a transform's parent space, so it can
    drive that transform's `.translate` no matter where it sits in the hierarchy."""
    n = cmds.createNode("vectorProduct", name=name)
    cmds.setAttr(n + ".operation", 4)  # point-matrix product
    cmds.connectAttr(world_plug, n + ".input1")
    cmds.connectAttr(parent_inv_plug, n + ".matrix")
    return n + ".output"


def _scale_vp(a, s_plug, name):
    """Scale a 3-vector plug by a scalar plug."""
    n = cmds.createNode("multiplyDivide", name=name)
    cmds.connectAttr(a, n + ".input1")
    for ax in "XYZ":
        cmds.connectAttr(s_plug, n + ".input2" + ax)
    return n + ".output"


def _lerp_v(a, b, t_plug, name):
    """a + t*(b-a) for 3-vector plugs `a`,`b` and scalar plug `t`."""
    return _add(a, _scale_vp(_sub(b, a, name + "_d"), t_plug, name + "_wd"), name + "_l")


def _div_vp(a, s_plug, name):
    """Divide a 3-vector plug by a scalar plug."""
    n = cmds.createNode("multiplyDivide", name=name)
    cmds.setAttr(n + ".operation", 2)  # divide
    cmds.connectAttr(a, n + ".input1")
    for ax in "XYZ":
        cmds.connectAttr(s_plug, n + ".input2" + ax)
    return n + ".output"


def _add_tan_attrs(ctrl):
    cmds.addAttr(ctrl, longName="Auto", attributeType="double",
                 defaultValue=1.0, min=0.0, max=1.0, keyable=True)
    cmds.addAttr(ctrl, longName="Smooth", attributeType="double",
                 defaultValue=1.0, min=0.0, max=1.0, keyable=True)
    cmds.addAttr(ctrl, longName="Weight", attributeType="double",
                 defaultValue=1.0, min=0.0, max=3.0, keyable=True)


def _build_tangents(name, grp, cv_ctrls, cv_pos, cvs, rest_in, rest_out,
                    start_tension=2.0, tan_rest=1.0,
                    out_ctrl=None, in_ctrl=None, out_buf=None, in_buf=None,
                    rest_out_plug=None, rest_in_plug=None):
    """Native port of twistMultiTangent handle positions (open spline).

    Computes the smooth (half-angle Catmull-Rom, weighted) and linear tangents
    live, blends smooth<->linear by Smooth and auto<->manual by Auto. Each tangent
    control rides an auto-driven BUFFER so it tracks the computed tangent in Auto
    mode; the control's translate is the manual offset. The spline reads the
    control's world position, so a manual offset always bends it.

    If `out_ctrl`/`in_ctrl`/`out_buf`/`in_buf` are given (the shared control rig),
    they are used as-is; otherwise simple locator controls are created. Likewise
    `rest_out_plug`/`rest_in_plug` (the Auto=0 rest reference, world-space point
    plugs per CV) override the default straight +/- tan_rest along the CV's X.
    Returns (out_handle, in_handle, out_ctrl, in_ctrl, out_buf, in_buf,
    rest_off_out, rest_off_in).
    """
    n = len(cv_ctrls)
    shared = out_ctrl is not None
    if not shared:
        out_ctrl, in_ctrl = [None] * n, [None] * n
        out_buf, in_buf = [None] * n, [None] * n
    out_w, in_w = [None] * n, [None] * n
    out_s, in_s = [None] * n, [None] * n
    out_a, in_a = [None] * n, [None] * n
    man_out, man_in = [None] * n, [None] * n
    for i in range(n):
        if i < n - 1:
            if not shared:
                buf = cmds.createNode("transform", name="{}_outbuf{}".format(name, i), parent=grp)
                c = cmds.spaceLocator(name="{}_outtan{}".format(name, i))[0]
                c = cmds.ls(cmds.parent(c, buf)[0], long=True)[0]
                cmds.setAttr(c + ".translate", 0, 0, 0)
                _add_tan_attrs(c)
                out_buf[i], out_ctrl[i] = buf, c
            c = out_ctrl[i]
            out_w[i], out_s[i], out_a[i] = c + ".Weight", c + ".Smooth", c + ".Auto"
            man_out[i] = _decompose(c)
        if i > 0:
            if not shared:
                buf = cmds.createNode("transform", name="{}_inbuf{}".format(name, i), parent=grp)
                c = cmds.spaceLocator(name="{}_intan{}".format(name, i))[0]
                c = cmds.ls(cmds.parent(c, buf)[0], long=True)[0]
                cmds.setAttr(c + ".translate", 0, 0, 0)
                _add_tan_attrs(c)
                in_buf[i], in_ctrl[i] = buf, c
            c = in_ctrl[i]
            in_w[i], in_s[i], in_a[i] = c + ".Weight", c + ".Smooth", c + ".Auto"
            man_in[i] = _decompose(c)

    # legs: in-leg -> prev CV, out-leg -> next CV (shared length per segment)
    in_len = [None] * n
    out_len = [None] * n
    in_norm = [None] * n
    out_norm = [None] * n
    for i in range(1, n):
        leg = _sub(cv_pos[i - 1], cv_pos[i], "{}_inleg{}".format(name, i))
        in_len[i] = _len_v(leg, "{}_inlen{}".format(name, i))
        in_norm[i] = _norm_v(leg, "{}_innrm{}".format(name, i))
        out_len[i - 1] = in_len[i]
        out_norm[i - 1] = _scale(in_norm[i], -1.0, "{}_outnrm{}".format(name, i - 1))

    # per-CV handle-length coefficients  len*weight/3
    in_coef = [None] * n
    out_coef = [None] * n
    in_smooth = [None] * n
    out_smooth = [None] * n
    for i in range(1, n - 1):
        in_coef[i] = _scale1(_mul1(in_len[i], in_w[i], "{}_icw{}".format(name, i)),
                             1.0 / 3.0, "{}_ic{}".format(name, i))
        out_coef[i] = _scale1(_mul1(out_len[i], out_w[i], "{}_ocw{}".format(name, i)),
                              1.0 / 3.0, "{}_oc{}".format(name, i))
        # half-angle bisector of the travel directions = normalize(outNorm - inNorm).
        # Identical to the C++ cross-of-cross bisector everywhere it's defined (to
        # ~1e-15) but robust for straight/collinear legs -- it yields the straight-
        # through tangent instead of dividing by a zero binormal. (Only an exact
        # fold, inNorm == outNorm, stays degenerate; that's a pathological cusp.)
        tan = _norm_v(_sub(out_norm[i], in_norm[i], "{}_ts{}".format(name, i)),
                      "{}_td{}".format(name, i))
        in_smooth[i] = _scale_vp(_scale(tan, -1.0, "{}_nt{}".format(name, i)),
                                 in_coef[i], "{}_ism{}".format(name, i))
        out_smooth[i] = _scale_vp(tan, out_coef[i], "{}_osm{}".format(name, i))

    # endpoint smooth tangents (open): extrapolate from the neighbour
    out_smooth[0] = _scale_vp(
        _sub(_add(cv_pos[1], _scale(in_smooth[1], start_tension, name + "_e0t"), name + "_e0a"),
             cv_pos[0], name + "_e0s"),
        _scale1(out_w[0], 0.5, name + "_e0c"), name + "_osm0")
    in_smooth[n - 1] = _scale_vp(
        _sub(_add(cv_pos[n - 2], _scale(out_smooth[n - 2], start_tension, name + "_ent"), name + "_ena"),
             cv_pos[n - 1], name + "_ens"),
        _scale1(in_w[n - 1], 0.5, name + "_enc"), "{}_ism{}".format(name, n - 1))

    # linear tangents
    in_linear = [None] * n
    out_linear = [None] * n
    for i in range(1, n - 1):
        in_dir = _norm_v(_sub(_add(cv_pos[i - 1],
                                   _scale_vp(out_smooth[i - 1], out_s[i - 1], "{}_ils{}".format(name, i)),
                                   "{}_ila{}".format(name, i)),
                              cv_pos[i], "{}_ild{}".format(name, i)), "{}_iln{}".format(name, i))
        out_dir = _norm_v(_sub(_add(cv_pos[i + 1],
                                    _scale_vp(in_smooth[i + 1], in_s[i + 1], "{}_ols{}".format(name, i)),
                                    "{}_ola{}".format(name, i)),
                               cv_pos[i], "{}_old{}".format(name, i)), "{}_oln{}".format(name, i))
        in_linear[i] = _scale_vp(in_dir, in_coef[i], "{}_ilin{}".format(name, i))
        out_linear[i] = _scale_vp(out_dir, out_coef[i], "{}_olin{}".format(name, i))
    out_linear[0] = _div_vp(
        _sub(_add(cv_pos[1], _scale_vp(in_smooth[1], in_s[1], name + "_l0s"), name + "_l0a"),
             cv_pos[0], name + "_l0d"),
        _sub_cp(3.0, in_s[1], name + "_l0n"), name + "_olin0")
    in_linear[n - 1] = _div_vp(
        _sub(_add(cv_pos[n - 2], _scale_vp(out_smooth[n - 2], out_s[n - 2], name + "_lns"), name + "_lna"),
             cv_pos[n - 1], name + "_lnd"),
        _sub_cp(3.0, out_s[n - 2], name + "_lnn"), "{}_ilin{}".format(name, n - 1))

    # blend smooth<->linear (Smooth) then auto<->manual (Auto), add to CV
    out_handle = [None] * n
    in_handle = [None] * n
    # Like the real rig: the buffer the control rides blends between the live auto
    # handle (Auto=1) and a frozen REST handle (Auto=0); the spline reads the
    # CONTROL's world position, so the control's own offset always bends the spline
    # (even at Auto=1). rest offsets follow the CV (added to its live position).
    rest_off_out = [None] * n
    rest_off_in = [None] * n
    # The rest reference (Auto=0 target) is a straight +/- tan_rest along the CV's
    # local X, exactly like the C++ rig's rest buffer -- so Auto blends the buffer
    # between that straight rest and the live auto tangent.
    for i in range(n):
        if i < n - 1:
            auto_vec = _lerp_v(out_linear[i], out_smooth[i], out_s[i], "{}_oav{}".format(name, i))
            auto_h = _add(cv_pos[i], auto_vec, "{}_oah{}".format(name, i))
            rest_off_out[i] = [tan_rest, 0.0, 0.0]
            rest_h = (rest_out_plug[i] if rest_out_plug is not None else
                      _point_mat([tan_rest, 0.0, 0.0], cv_ctrls[i] + ".worldMatrix[0]",
                                 "{}_orh{}".format(name, i)))
            buf = _lerp_v(rest_h, auto_h, out_a[i], "{}_obp{}".format(name, i))
            # drive the buffer's translate in its own parent space (works under any
            # hierarchy: grouped flat for native, nested for the shared control rig)
            cmds.connectAttr(_world_to_local(buf, out_buf[i] + ".parentInverseMatrix[0]",
                                             "{}_ob2l{}".format(name, i)),
                             out_buf[i] + ".translate")
            out_handle[i] = man_out[i]   # spline follows the control's world position
        if i > 0:
            auto_vec = _lerp_v(in_linear[i], in_smooth[i], in_s[i], "{}_iav{}".format(name, i))
            auto_h = _add(cv_pos[i], auto_vec, "{}_iah{}".format(name, i))
            rest_off_in[i] = [-tan_rest, 0.0, 0.0]
            rest_h = (rest_in_plug[i] if rest_in_plug is not None else
                      _point_mat([-tan_rest, 0.0, 0.0], cv_ctrls[i] + ".worldMatrix[0]",
                                 "{}_irh{}".format(name, i)))
            buf = _lerp_v(rest_h, auto_h, in_a[i], "{}_ibp{}".format(name, i))
            cmds.connectAttr(_world_to_local(buf, in_buf[i] + ".parentInverseMatrix[0]",
                                             "{}_ib2l{}".format(name, i)),
                             in_buf[i] + ".translate")
            in_handle[i] = man_in[i]
    return (out_handle, in_handle, out_ctrl, in_ctrl, out_buf, in_buf,
            rest_off_out, rest_off_in)


# ---- RMF transport node clusters (Stage 2) --------------------------------

def _poci(curve_shape, param, name):
    p = cmds.createNode("pointOnCurveInfo", name=name)
    cmds.connectAttr(curve_shape + ".worldSpace[0]", p + ".inputCurve")
    cmds.setAttr(p + ".turnOnPercentage", 0)
    cmds.setAttr(p + ".parameter", param)
    return p  # .position, .normalizedTangent


def _poci_live(curve_shape, param_plug, name):
    """Like _poci but the parameter is a *live* plug (pin-driven)."""
    p = cmds.createNode("pointOnCurveInfo", name=name)
    cmds.connectAttr(curve_shape + ".worldSpace[0]", p + ".inputCurve")
    cmds.setAttr(p + ".turnOnPercentage", 0)
    cmds.connectAttr(param_plug, p + ".parameter")
    return p


def _motionpath(curve_shape, frac_plug, name):
    """Point at an arc-length FRACTION of the curve (motionPath, fractionMode)."""
    mp = cmds.createNode("motionPath", name=name)
    cmds.connectAttr(curve_shape + ".worldSpace[0]", mp + ".geometryPath")
    cmds.setAttr(mp + ".fractionMode", 1)
    cmds.connectAttr(frac_plug, mp + ".uValue")
    return mp  # .allCoordinates


def _nearest_param(curve_shape, pos_plug, name):
    """Curve parameter of the on-curve point nearest pos_plug."""
    npc = cmds.createNode("nearestPointOnCurve", name=name)
    cmds.connectAttr(curve_shape + ".worldSpace[0]", npc + ".inputCurve")
    cmds.connectAttr(pos_plug, npc + ".inPosition")
    return npc + ".parameter"


def _world_y(ctrl, name):
    """World-space Y axis (up) of a transform as a direction vector plug."""
    n = cmds.createNode("vectorProduct", name=name)
    cmds.setAttr(n + ".operation", 3)  # vector-matrix product (ignores translation)
    cmds.setAttr(n + ".input1", 0.0, 1.0, 0.0)
    cmds.connectAttr(ctrl + ".worldMatrix[0]", n + ".matrix")
    return n + ".output"


def _reproject(vec, tan, name):
    """Normalize(vec - (vec.tan) tan): make `vec` perpendicular to unit `tan`."""
    dot = cmds.createNode("vectorProduct", name=name + "_dot")
    cmds.setAttr(dot + ".operation", 1)  # dot
    cmds.connectAttr(vec, dot + ".input1")
    cmds.connectAttr(tan, dot + ".input2")
    sc = cmds.createNode("multiplyDivide", name=name + "_sc")
    cmds.connectAttr(tan, sc + ".input1")
    for ax in "XYZ":
        cmds.connectAttr(dot + ".outputX", sc + ".input2" + ax)
    sub = cmds.createNode("plusMinusAverage", name=name + "_sub")
    cmds.setAttr(sub + ".operation", 2)
    cmds.connectAttr(vec, sub + ".input3D[0]")
    cmds.connectAttr(sc + ".output", sub + ".input3D[1]")
    nrm = cmds.createNode("vectorProduct", name=name + "_nrm")
    cmds.setAttr(nrm + ".operation", 0)  # no-op, just normalize
    cmds.setAttr(nrm + ".normalizeOutput", 1)
    cmds.connectAttr(sub + ".output3D", nrm + ".input1")
    return nrm + ".output"


def _transport(up_prev, t_prev, t_cur, name):
    """Parallel-transport up_prev across the tangent change, then reproject."""
    ab = cmds.createNode("angleBetween", name=name + "_ab")
    cmds.connectAttr(t_prev, ab + ".vector1")
    cmds.connectAttr(t_cur, ab + ".vector2")
    aaq = cmds.createNode("axisAngleToQuat", name=name + "_aaq")
    # angleBetween.axis is NOT unit length; axisAngleToQuat assumes a unit axis,
    # so feeding it raw builds a non-unit quaternion -> a small per-step rotation
    # error that accumulates with the curve's total winding. Normalize it first.
    cmds.connectAttr(_norm_v(ab + ".axis", name + "_axn"), aaq + ".inputAxis")
    cmds.connectAttr(ab + ".angle", aaq + ".inputAngle")
    cm = cmds.createNode("composeMatrix", name=name + "_cm")
    cmds.setAttr(cm + ".useEulerRotation", 0)
    cmds.connectAttr(aaq + ".outputQuat", cm + ".inputQuat")
    pmm = cmds.createNode("vectorProduct", name=name + "_rot")
    cmds.setAttr(pmm + ".operation", 3)  # vector-matrix product
    cmds.connectAttr(up_prev, pmm + ".input1")
    cmds.connectAttr(cm + ".outputMatrix", pmm + ".matrix")
    return _reproject(pmm + ".output", t_cur, name + "_rp")


def _dot(a, b, name):
    """Dot product of two 3-vector plugs (scalar)."""
    n = cmds.createNode("vectorProduct", name=name)
    cmds.setAttr(n + ".operation", 1)  # dot
    cmds.connectAttr(a, n + ".input1")
    cmds.connectAttr(b, n + ".input2")
    return n + ".outputX"


def _const_div(num, denom_plug, name):
    """const / plug (1D)."""
    n = cmds.createNode("multiplyDivide", name=name)
    cmds.setAttr(n + ".operation", 2)  # divide
    cmds.setAttr(n + ".input1X", num)
    cmds.connectAttr(denom_plug, n + ".input2X")
    return n + ".outputX"


def _double_reflect(up, p0, p1, t0, t1, name):
    """One double-reflection RMF step (Wang et al.), the method the C++ kernel
    uses. Reflects the frame across the bisecting plane of the two sample points,
    then across the plane that maps t0 onto t1 -- a more stable transport than the
    angleBetween rotation, and the one needed to match the C++ rig exactly.
    """
    v1 = _sub(p1, p0, name + "_v1")
    c1 = _const_div(2.0, _dot(v1, v1, name + "_d11"), name + "_c1")  # 2/|v1|^2
    # reflect tangent: t_li = t0 - v1 * (c1 * dot(v1, t0))
    t_li = _sub(t0, _scale_vp(v1, _mul1(c1, _dot(v1, t0, name + "_dvt"), name + "_c1t"),
                              name + "_v1t"), name + "_tli")
    v2 = _sub(t1, t_li, name + "_v2")
    c2 = _const_div(2.0, _dot(v2, v2, name + "_d22"), name + "_c2")
    # reflect normal twice: r_li = up - v1*(c1*dot(v1,up)); up1 = r_li - v2*(c2*dot(v2,r_li))
    r_li = _sub(up, _scale_vp(v1, _mul1(c1, _dot(v1, up, name + "_dvu"), name + "_c1u"),
                              name + "_v1u"), name + "_rli")
    up1 = _sub(r_li, _scale_vp(v2, _mul1(c2, _dot(v2, r_li, name + "_dvr"), name + "_c2r"),
                               name + "_v2r"), name + "_up1")
    return _reproject(up1, t1, name + "_rp")


def _build_frame(tan, up, pos, joint, name):
    """rows X=tan, Y=up, Z=cross(tan,up); translation=pos -> drive joint.

    tan/up/pos are parent double3 plugs (we append X/Y/Z for the components).
    """
    z = cmds.createNode("vectorProduct", name=name + "_bin")
    cmds.setAttr(z + ".operation", 2)  # cross
    cmds.setAttr(z + ".normalizeOutput", 1)
    cmds.connectAttr(tan, z + ".input1")
    cmds.connectAttr(up, z + ".input2")

    fbf = cmds.createNode("fourByFourMatrix", name=name + "_mtx")
    for row, plug in ((0, tan), (1, up), (2, z + ".output"), (3, pos)):
        for col, ax in enumerate("XYZ"):
            cmds.connectAttr(plug + ax, "{}.in{}{}".format(fbf, row, col))

    dm = cmds.createNode("decomposeMatrix", name=name + "_dec")
    cmds.connectAttr(fbf + ".output", dm + ".inputMatrix")
    cmds.setAttr(joint + ".jointOrient", 0, 0, 0)
    cmds.connectAttr(dm + ".outputTranslate", joint + ".translate")
    cmds.connectAttr(dm + ".outputRotate", joint + ".rotate")


# ---- scalar / twist helpers (Stage 3) -------------------------------------

def _alen(curve_shape, param, name):
    n = cmds.createNode("arcLengthDimension", name=name + "Shape")
    cmds.connectAttr(curve_shape + ".worldSpace[0]", n + ".nurbsGeometry")
    cmds.setAttr(n + ".uParamValue", param)
    return n + ".arcLength"


def _sub1(a, b, name):
    n = cmds.createNode("plusMinusAverage", name=name)
    cmds.setAttr(n + ".operation", 2)
    cmds.connectAttr(a, n + ".input1D[0]")
    cmds.connectAttr(b, n + ".input1D[1]")
    return n + ".output1D"


def _mul1(a, b, name, divide=False):
    n = cmds.createNode("multiplyDivide", name=name)
    cmds.setAttr(n + ".operation", 2 if divide else 1)
    cmds.connectAttr(a, n + ".input1X")
    cmds.connectAttr(b, n + ".input2X")
    return n + ".outputX"


def _add1(a, b, name):
    n = cmds.createNode("plusMinusAverage", name=name)
    cmds.setAttr(n + ".operation", 1)  # sum
    cmds.connectAttr(a, n + ".input1D[0]")
    cmds.connectAttr(b, n + ".input1D[1]")
    return n + ".output1D"


def _lerp1(a, b, w, name):
    """a + w*(b-a)."""
    return _add1(a, _mul1(_sub1(b, a, name + "_d"), w, name + "_wd"), name + "_l")


def _sub_cp(c, plug, name):
    """const - plug (1D)."""
    n = cmds.createNode("plusMinusAverage", name=name)
    cmds.setAttr(n + ".operation", 2)
    cmds.setAttr(n + ".input1D[0]", c)
    cmds.connectAttr(plug, n + ".input1D[1]")
    return n + ".output1D"


def _add_pc(plug, c, name):
    """plug + const (1D)."""
    n = cmds.createNode("plusMinusAverage", name=name)
    cmds.setAttr(n + ".operation", 1)
    cmds.connectAttr(plug, n + ".input1D[0]")
    cmds.setAttr(n + ".input1D[1]", c)
    return n + ".output1D"


def _scale1(plug, c, name):
    """plug * const (1D)."""
    n = cmds.createNode("multiplyDivide", name=name)
    cmds.setAttr(n + ".operation", 1)
    cmds.connectAttr(plug, n + ".input1X")
    cmds.setAttr(n + ".input2X", c)
    return n + ".outputX"


def _clamp01(plug, name):
    """clamp(plug, 0, 1) (1D)."""
    n = cmds.createNode("clamp", name=name)
    cmds.setAttr(n + ".minR", 0.0)
    cmds.setAttr(n + ".maxR", 1.0)
    cmds.connectAttr(plug, n + ".inputR")
    return n + ".outputR"


def _sumN(plugs, name):
    """Sum of a list of 1D plugs."""
    n = cmds.createNode("plusMinusAverage", name=name)
    cmds.setAttr(n + ".operation", 1)
    for i, p in enumerate(plugs):
        cmds.connectAttr(p, "{}.input1D[{}]".format(n, i))
    return n + ".output1D"


def _solve_param_remap(name, pin, pp, arc):
    """Native Thomas solve of solveParamMatrix -> a live remap-param plug per CV.

    `pin[i]`/`pp[i]` are the Pin/PinParam attr plugs and `arc[i]` the cumulative
    arc-length plugs. The row equation rearranges to
        x[i] = Pin[i]*PinParam[i] + (1-Pin[i])*((1-A)x[i-1] + A x[i+1])
    so any *fractional* Pin is exact -- this runs the full tridiagonal solve live
    (every diagonal is -1, so the Thomas recurrence is a short node chain).
    """
    n = len(pin)
    oml = [_sub_cp(1.0, pin[i], "{}_oml{}".format(name, i)) for i in range(n)]  # 1-Pin
    sub = [None] * n
    sup = [None] * n
    res = [None] * n
    # start row: [0, -1, 1-lv0],  res = -lv0*rv0 + (1-lv0)(cv1-cv0)
    sup[0] = oml[0]
    res[0] = _add1(_scale1(_mul1(pin[0], pp[0], name + "_r0a"), -1.0, name + "_r0b"),
                   _mul1(oml[0], _sub1(arc[1], arc[0], name + "_r0c"), name + "_r0d"),
                   name + "_res0")
    # interior rows
    for i in range(1, n - 1):
        A = _mul1(_sub1(arc[i], arc[i - 1], "{}_An{}".format(name, i)),
                  _sub1(arc[i + 1], arc[i - 1], "{}_Ad{}".format(name, i)),
                  "{}_A{}".format(name, i), divide=True)
        sub[i] = _mul1(oml[i], _sub_cp(1.0, A, "{}_omA{}".format(name, i)),
                       "{}_sub{}".format(name, i))
        sup[i] = _mul1(oml[i], A, "{}_sup{}".format(name, i))
        res[i] = _scale1(_mul1(pin[i], pp[i], "{}_ria{}".format(name, i)), -1.0,
                         "{}_res{}".format(name, i))
    # end row: [1-lve, -1, 0],  res = -lve*rve - (1-lve)(cve-cv_{e-1})
    e = n - 1
    sub[e] = oml[e]
    res[e] = _sub1(_scale1(_mul1(pin[e], pp[e], name + "_rea"), -1.0, name + "_reb"),
                   _mul1(oml[e], _sub1(arc[e], arc[e - 1], name + "_rec"), name + "_red"),
                   name + "_rese")
    # Thomas forward sweep (diag == -1)
    c = [None] * n
    d = [None] * n
    c[0] = _scale1(sup[0], -1.0, name + "_c0")
    d[0] = _scale1(res[0], -1.0, name + "_d0")
    for i in range(1, n):
        denom = _sub_cp(-1.0, _mul1(sub[i], c[i - 1], "{}_sc{}".format(name, i)),
                        "{}_den{}".format(name, i))
        if i < n - 1:
            c[i] = _mul1(sup[i], denom, "{}_c{}".format(name, i), divide=True)
        num = _sub1(res[i], _mul1(sub[i], d[i - 1], "{}_sd{}".format(name, i)),
                    "{}_num{}".format(name, i))
        d[i] = _mul1(num, denom, "{}_d{}".format(name, i), divide=True)
    # back substitution
    x = [None] * n
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = _sub1(d[i], _mul1(c[i], x[i + 1], "{}_bx{}".format(name, i)),
                     "{}_x{}".format(name, i))
    return x


def _solve_twist_param(name, lock, val, arc):
    """Native Thomas solve of solveTwistParamMatrix -> a live twist value per CV.

    Same tridiagonal matrix as the Pin solve but RHS = UseTwist*Twist: twist-pinned
    CVs hold their Twist, unpinned CVs FLOAT (arc-interpolated between pins) instead
    of being kinked to zero. Exact for fractional UseTwist (matches the kernel).
    `lock[k]`=UseTwist, `val[k]`=Twist, `arc[k]`=cumulative arc length (all plugs).
    """
    n = len(lock)
    oml = [_sub_cp(1.0, lock[i], "{}_oml{}".format(name, i)) for i in range(n)]
    sub = [None] * n
    sup = [None] * n
    res = [None] * n
    # RHS is -(UseTwist*Twist): the kernel's solve returns the NEGATED twist (a
    # pinned CV holds -Twist) and flips it back downstream; negating the RHS makes
    # this solve hold +Twist directly, matching the native _apply_twist convention.
    sup[0] = oml[0]
    res[0] = _scale1(_mul1(lock[0], val[0], name + "_rm0"), -1.0, name + "_res0")
    for i in range(1, n - 1):
        A = _mul1(_sub1(arc[i], arc[i - 1], "{}_An{}".format(name, i)),
                  _sub1(arc[i + 1], arc[i - 1], "{}_Ad{}".format(name, i)),
                  "{}_A{}".format(name, i), divide=True)
        sub[i] = _mul1(oml[i], _sub_cp(1.0, A, "{}_omA{}".format(name, i)),
                       "{}_sub{}".format(name, i))
        sup[i] = _mul1(oml[i], A, "{}_sup{}".format(name, i))
        res[i] = _scale1(_mul1(lock[i], val[i], "{}_rm{}".format(name, i)), -1.0,
                         "{}_res{}".format(name, i))
    e = n - 1
    sub[e] = oml[e]
    res[e] = _scale1(_mul1(lock[e], val[e], name + "_rme"), -1.0, name + "_rese")
    c = [None] * n
    d = [None] * n
    c[0] = _scale1(sup[0], -1.0, name + "_c0")
    d[0] = _scale1(res[0], -1.0, name + "_d0")
    for i in range(1, n):
        denom = _sub_cp(-1.0, _mul1(sub[i], c[i - 1], "{}_sc{}".format(name, i)),
                        "{}_den{}".format(name, i))
        if i < n - 1:
            c[i] = _mul1(sup[i], denom, "{}_c{}".format(name, i), divide=True)
        num = _sub1(res[i], _mul1(sub[i], d[i - 1], "{}_sd{}".format(name, i)),
                    "{}_num{}".format(name, i))
        d[i] = _mul1(num, denom, "{}_d{}".format(name, i), divide=True)
    x = [None] * n
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = _sub1(d[i], _mul1(c[i], x[i + 1], "{}_bx{}".format(name, i)),
                     "{}_x{}".format(name, i))
    return x


def _signed_angle(a, b, axis, name):
    """Signed angle from unit `a` to unit `b` measured around unit `axis`.

    Magnitude from angleBetween; sign from sign(cross(a,b) . axis).
    """
    ab = cmds.createNode("angleBetween", name=name + "_ab")
    cmds.connectAttr(a, ab + ".vector1")
    cmds.connectAttr(b, ab + ".vector2")
    cr = cmds.createNode("vectorProduct", name=name + "_cr")
    cmds.setAttr(cr + ".operation", 2)  # cross
    cmds.connectAttr(a, cr + ".input1")
    cmds.connectAttr(b, cr + ".input2")
    dt = cmds.createNode("vectorProduct", name=name + "_dt")
    cmds.setAttr(dt + ".operation", 1)  # dot
    cmds.connectAttr(cr + ".output", dt + ".input1")
    cmds.connectAttr(axis, dt + ".input2")
    cond = cmds.createNode("condition", name=name + "_sgn")
    cmds.setAttr(cond + ".operation", 3)  # >=
    cmds.connectAttr(dt + ".outputX", cond + ".firstTerm")
    cmds.setAttr(cond + ".secondTerm", 0.0)
    cmds.setAttr(cond + ".colorIfTrueR", 1.0)
    cmds.setAttr(cond + ".colorIfFalseR", -1.0)
    return _mul1(ab + ".angle", cond + ".outColorR", name + "_s")


def _apply_twist(up, tan, angle, name):
    """Rotate unit `up` around unit `tan` by `angle` (radians plug)."""
    aaq = cmds.createNode("axisAngleToQuat", name=name + "_aaq")
    cmds.connectAttr(tan, aaq + ".inputAxis")
    cmds.connectAttr(angle, aaq + ".inputAngle")
    cm = cmds.createNode("composeMatrix", name=name + "_cm")
    cmds.setAttr(cm + ".useEulerRotation", 0)
    cmds.connectAttr(aaq + ".outputQuat", cm + ".inputQuat")
    vp = cmds.createNode("vectorProduct", name=name + "_rot")
    cmds.setAttr(vp + ".operation", 3)
    cmds.connectAttr(up, vp + ".input1")
    cmds.connectAttr(cm + ".outputMatrix", vp + ".matrix")
    return vp + ".output"


# ---- control attributes ---------------------------------------------------

def _add_cv_attrs(ctrl):
    cmds.addAttr(ctrl, longName="UseOrient", attributeType="double",
                 defaultValue=0.0, min=0.0, max=1.0, keyable=True)
    cmds.addAttr(ctrl, longName="Pin", attributeType="double",
                 defaultValue=0.0, min=0.0, max=1.0, keyable=True)
    cmds.addAttr(ctrl, longName="PinParam", attributeType="double",
                 defaultValue=0.0, keyable=True)


def _add_twist_attrs(ctrl):
    """Per-CV twist control: rotateX injects twist; UseTwist is its lock weight."""
    cmds.addAttr(ctrl, longName="UseTwist", attributeType="double",
                 defaultValue=1.0, min=0.0, max=1.0, keyable=True)


# ---- Stage 1 build --------------------------------------------------------

def build_native_spline(cv_positions, num_joints, spread=3.0, name="nativeTS",
                        samples_per_interval=20, pins=None, orient_cvs=None,
                        tan_rest=None, controls=None):
    """Live curve from CV controls + RMF-oriented joints + per-CV control attrs.

    Position via the live degree-3 curve, orientation via a double-reflection RMF.
    Returns a dict of node names.

    If `controls` (a dict from the shared production control rig) is given, the
    native node graph is wired onto those controls instead of creating its own --
    it must provide grp, cv_ctrls, twist_ctrl, out_ctrl, in_ctrl, out_buf, in_buf,
    rest_out, rest_in. Otherwise simple locator controls are created (standalone).
    """
    # matrixNodes (decompose/compose/pointMatrixMult/fourByFourMatrix) and
    # quatNodes (axisAngleToQuat) ship WITH Maya and auto-load on scene open --
    # no external files for the animator, just standard Maya requirements.
    for _plug in ("matrixNodes", "quatNodes"):
        if not cmds.pluginInfo(_plug, q=True, loaded=True):
            cmds.loadPlugin(_plug, quiet=True)

    cvs = [list(p) for p in cv_positions]
    n = len(cvs)
    nseg = n - 1
    # rest (default) bezier handle positions from the faithful tangent reference
    rest_in, rest_out = multi_tangent_handles(cvs)

    if controls is None:
        grp = cmds.createNode("transform", name=name + "_grp")
        # CV control transforms with the per-CV attributes, each with a child Twist
        # control (its rotateX injects twist at that CV, like the real rig).
        cv_ctrls = []
        twist_ctrls = []
        for i in range(n):
            c = cmds.spaceLocator(name="{}_cv{}".format(name, i))[0]
            # full DAG path so a same-named rebuild in one scene stays unambiguous
            c = cmds.ls(cmds.parent(c, grp)[0], long=True)[0]
            cmds.xform(c, worldSpace=True, translation=cvs[i])
            _add_cv_attrs(c)
            cv_ctrls.append(c)
            tw = cmds.spaceLocator(name="{}_cv{}_twist".format(name, i))[0]
            tw = cmds.ls(cmds.parent(tw, c)[0], long=True)[0]
            cmds.setAttr(tw + ".translate", 0, 0, 0)
            _add_twist_attrs(tw)
            twist_ctrls.append(tw)
        tan_kwargs = {}
    else:
        grp = controls["grp"]
        cv_ctrls = controls["cv_ctrls"]
        twist_ctrls = controls["twist_ctrl"]
        tan_kwargs = dict(out_ctrl=controls["out_ctrl"], in_ctrl=controls["in_ctrl"],
                          out_buf=controls["out_buf"], in_buf=controls["in_buf"],
                          rest_out_plug=controls["rest_out"],
                          rest_in_plug=controls["rest_in"])

    curve_tfm = cmds.createNode("transform", name=name + "_curve", parent=grp)

    # Default pin pattern (matches the real rig): twist@CV0, orient@first+last,
    # and the endpoints anchor the param range (Pin is a live 0..1 blend).
    cmds.setAttr(twist_ctrls[0] + ".UseTwist", 1.0)
    # orient-locked CVs (default first+last; pass orient_cvs=[0] to match the C++
    # builder's CV0-only default for parity).
    for k in (orient_cvs if orient_cvs is not None else [0, n - 1]):
        cmds.setAttr(cv_ctrls[k] + ".UseOrient", 1.0)
    cmds.setAttr(cv_ctrls[0] + ".Pin", 1.0)
    cmds.setAttr(cv_ctrls[-1] + ".Pin", 1.0)
    # Extra position pins requested at build time (default the slider to fully on).
    for k in (pins or []):
        cmds.setAttr(cv_ctrls[k] + ".Pin", 1.0)

    cv_pos = [_decompose(c) for c in cv_ctrls]

    # Degree-3 bezier-form curve: control points [cv0, out0, in1, cv1, out1, ...].
    # Default handle positions come from the faithful tangent reference.
    init_pts = []
    for k in range(nseg):
        init_pts.extend([cvs[k], rest_out[k], rest_in[k + 1]])
    init_pts.append(cvs[-1])
    # Bezier knot vector: internal joints have multiplicity 3.
    knots = ([0.0, 0.0, 0.0]
             + [float(j) for j in range(1, nseg) for _ in range(3)]
             + [float(nseg)] * 3)

    tmp = cmds.curve(degree=3, point=init_pts, knot=knots)
    shp = cmds.listRelatives(tmp, shapes=True, fullPath=True)[0]
    cmds.parent(shp, curve_tfm, shape=True, relative=True)
    cmds.delete(tmp)
    curve_shape = cmds.listRelatives(curve_tfm, shapes=True)[0]

    # ---- Stage 6: tangent controls (in/out per CV, with Auto/Smooth/Weight).
    # Faithful port of twistMultiTangent: half-angle Catmull-Rom smooth tangents,
    # smooth<->linear blend, manual override, all live. Everything downstream rides
    # the curve, so no other stage changes.
    (out_handle, in_handle, out_ctrl, in_ctrl, out_buf, in_buf,
     rest_off_out, rest_off_in) = _build_tangents(
        name, grp, cv_ctrls, cv_pos, cvs, rest_in, rest_out,
        tan_rest=(spread if tan_rest is None else tan_rest), **tan_kwargs)

    # Drive every control point from the live CV / tangent-handle plugs.
    for k in range(nseg):
        cmds.connectAttr(cv_pos[k], "{}.controlPoints[{}]".format(curve_shape, 3 * k))
        cmds.connectAttr(out_handle[k], "{}.controlPoints[{}]".format(curve_shape, 3 * k + 1))
        cmds.connectAttr(in_handle[k + 1], "{}.controlPoints[{}]".format(curve_shape, 3 * k + 2))
    cmds.connectAttr(cv_pos[-1], "{}.controlPoints[{}]".format(curve_shape, 3 * nseg))

    # ---- RMF transport chain, sampled densely by curve PARAMETER (a CV at each
    # integer param, plus `samples_per_interval` fill points per segment). Joints
    # are NOT samples -- they read the upCurve at their own (possibly pinned) param.
    sample_params = []
    for k in range(nseg):
        for s in range(samples_per_interval):
            sample_params.append(k + s / float(samples_per_interval))
    sample_params.append(float(nseg))
    cv_idx = {k: k * samples_per_interval for k in range(nseg + 1)}

    pocis = [_poci(curve_shape, p, "{}_poci{}".format(name, i))
             for i, p in enumerate(sample_params)]
    tans = [pc + ".normalizedTangent" for pc in pocis]
    # At a non-G1 (kinked) CV -- Auto<1, a moved handle, etc. -- the curve's tangent
    # is one-sided and pointOnCurveInfo at the knot is ambiguous. Use the true
    # segment-START tangent (out_handle - CV) at each interior CV sample so the
    # transport reprojects across the kink exactly like the C++ per-segment RMF.
    # Smooth (G1) CVs are unaffected (in and out tangents agree there).
    for k in range(1, nseg):
        tans[cv_idx[k]] = _norm_v(_sub(out_handle[k], cv_pos[k], "{}_cvtd{}".format(name, k)),
                                  "{}_cvt{}".format(name, k))

    # anchor up = CV0 control Y, reprojected perpendicular to the start tangent
    ups = [None] * len(pocis)
    ups[0] = _reproject(_world_y(cv_ctrls[0], name + "_anchorY"), tans[0], name + "_anchor")
    for i in range(1, len(pocis)):
        ups[i] = _double_reflect(ups[i - 1], pocis[i - 1] + ".position", pocis[i] + ".position",
                                 tans[i - 1], tans[i], "{}_dr{}".format(name, i))

    # Stage 3: per-CV twist. Solve solveTwistParamMatrix (live) so twist-PINNED CVs
    # (UseTwist) hold their Twist and unpinned CVs FLOAT between pins -- then the
    # solved per-CV twist is interpolated by live arc length between CVs.
    arc_cv = [_alen(curve_shape, float(k), "{}_alenCV{}".format(name, k))
              for k in range(nseg + 1)]
    twist_val = _solve_twist_param(name + "_tw",
                                   [twist_ctrls[k] + ".UseTwist" for k in range(n)],
                                   [twist_ctrls[k] + ".rotateX" for k in range(n)], arc_cv)

    # Stage 4: useOrient. Residual at each orient-pinned CV (angle between the RMF
    # and the control's up, around the tangent) * UseOrient, distributed linearly
    # by arc length between consecutive orient pins. Pin set read at build time.
    orient_active = [k for k in range(n) if cmds.getAttr(cv_ctrls[k] + ".UseOrient") >= 0.5]
    orient_val = {}
    for k in orient_active:
        ci = cv_idx[k]
        cup = _reproject(_world_y(cv_ctrls[k], "{}_oy{}".format(name, k)),
                         tans[ci], "{}_orp{}".format(name, k))
        resid = _signed_angle(ups[ci], cup, tans[ci], "{}_res{}".format(name, k))
        orient_val[k] = _mul1(resid, cv_ctrls[k] + ".UseOrient", "{}_oval{}".format(name, k))

    # ---- Final up per sample = RMF up rotated by (twist + orient) at that sample.
    final_up = [None] * len(sample_params)
    for i, p_i in enumerate(sample_params):
        k = min(int(p_i + 1e-9), nseg - 1)  # segment = floor(param), clamped
        on_cv = abs(p_i - round(p_i)) < 1e-9
        arc_i = arc_cv[int(round(p_i))] if on_cv else \
            _alen(curve_shape, p_i, "{}_alenS{}".format(name, i))
        w = _mul1(_sub1(arc_i, arc_cv[k], "{}_swn{}".format(name, i)),
                  _sub1(arc_cv[k + 1], arc_cv[k], "{}_swd{}".format(name, i)),
                  "{}_sw{}".format(name, i), divide=True)
        angle = _lerp1(twist_val[k], twist_val[k + 1], w, "{}_stw{}".format(name, i))
        if orient_active:
            lo = max([a for a in orient_active if a <= p_i], default=orient_active[0])
            hi = min([a for a in orient_active if a >= p_i], default=orient_active[-1])
            if lo == hi:
                o_tw = orient_val[lo]
            else:
                f = _mul1(_sub1(arc_i, arc_cv[lo], "{}_sown{}".format(name, i)),
                          _sub1(arc_cv[hi], arc_cv[lo], "{}_sowd{}".format(name, i)),
                          "{}_sow{}".format(name, i), divide=True)
                o_tw = _lerp1(orient_val[lo], orient_val[hi], f, "{}_sotw{}".format(name, i))
            angle = _add1(angle, o_tw, "{}_stot{}".format(name, i))
        final_up[i] = _apply_twist(ups[i], tans[i], angle, "{}_satw{}".format(name, i))

    # ---- upCurve: a degree-1 curve whose control points ARE the final up-vectors,
    # so a joint at any (possibly pinned) param reads its interpolated frame up by
    # sampling -- the curve itself does the interpolation (no per-joint indexing).
    up_tfm = cmds.createNode("transform", name=name + "_upCurveT", parent=grp)
    tmp2 = cmds.curve(degree=1,
                      point=[[0.0, float(i), 0.0] for i in range(len(sample_params))])
    shp2 = cmds.listRelatives(tmp2, shapes=True, fullPath=True)[0]
    cmds.parent(shp2, up_tfm, shape=True, relative=True)
    cmds.delete(tmp2)
    up_curve = cmds.listRelatives(up_tfm, shapes=True)[0]
    for i in range(len(sample_params)):
        cmds.connectAttr(final_up[i], "{}.controlPoints[{}]".format(up_curve, i))

    # default PinParam = each CV's curve param (its position in the param map)
    for k in range(n):
        cmds.setAttr(cv_ctrls[k] + ".PinParam", float(k))

    # ---- Stage 5: Pin. The param map (remap) gives each CV's param: a PINNED CV
    # holds its PinParam; an UNPINNED CV floats to its arc-length position. This is
    # the full solveParamMatrix, run live as a native Thomas-solve node chain, so
    # Pin is a *live* 0..1 blend (no rebuild to toggle) and exact at every value.
    remap = _solve_param_remap(name + "_pin",
                               [cv_ctrls[k] + ".Pin" for k in range(n)],
                               [cv_ctrls[k] + ".PinParam" for k in range(n)], arc_cv)

    # Rest remap via the kernel solver -- used ONLY to assign each joint the segment
    # its rider param falls in. Valid as long as pins keep that param inside the
    # segment (true for moderate pinning).
    from ..solve import solve_param_matrix
    arc_rest = [cmds.getAttr(a) for a in arc_cv]
    pin_rest = [cmds.getAttr(cv_ctrls[k] + ".Pin") for k in range(n)]
    pp_rest = [cmds.getAttr(cv_ctrls[k] + ".PinParam") for k in range(n)]
    rest_remap = solve_param_matrix(pp_rest, arc_rest, pin_rest)
    pmin, pmax = rest_remap[0], rest_remap[-1]

    # ---- joints: distribute by ARC LENGTH within the pinned param range, exactly
    # like the C++ rider. The clamped per-segment fraction
    #   c_k = clamp((t - remap[k]) / (remap[k+1] - remap[k]), 0, 1)
    # serves double duty: sum(c_k) is the curve param, sum(c_k * segArc[k]) is the
    # target arc length. We place the joint at that arc length (motionPath), recover
    # its curve param (nearestPointOnCurve), and sample the orientation upCurve there.
    seg_arc = [_sub1(arc_cv[k + 1], arc_cv[k], "{}_segarc{}".format(name, k))
               for k in range(nseg)]
    total_arc = arc_cv[-1]
    joints = []
    for j in range(num_joints):
        t_j = pmin + (pmax - pmin) * j / (num_joints - 1.0) if num_joints > 1 else pmin
        cterms = []
        for k in range(nseg):
            inv = _mul1(_sub_cp(t_j, remap[k], "{}_jin{}_{}".format(name, j, k)),
                        _sub1(remap[k + 1], remap[k], "{}_jid{}_{}".format(name, j, k)),
                        "{}_ji{}_{}".format(name, j, k), divide=True)
            cterms.append(_clamp01(inv, "{}_jc{}_{}".format(name, j, k)))
        arc_terms = [_mul1(cterms[k], seg_arc[k], "{}_jat{}_{}".format(name, j, k))
                     for k in range(nseg)]
        target_arc = (_sumN(arc_terms, "{}_jta{}".format(name, j))
                      if nseg > 1 else arc_terms[0])
        frac = _mul1(target_arc, total_arc, "{}_jfr{}".format(name, j), divide=True)
        mp = _motionpath(curve_shape, frac, "{}_jmp{}".format(name, j))
        u_arc = _nearest_param(curve_shape, mp + ".allCoordinates", "{}_jnp{}".format(name, j))
        jp = _poci_live(curve_shape, u_arc, "{}_jp{}".format(name, j))
        up_par = _scale1(u_arc, float(samples_per_interval), "{}_jus{}".format(name, j))
        jup = _poci_live(up_curve, up_par, "{}_jup{}".format(name, j))
        up = _reproject(jup + ".position", jp + ".normalizedTangent", "{}_jrp{}".format(name, j))
        jt = cmds.createNode("joint", name="{}_jnt{}".format(name, j), parent=grp)
        _build_frame(jp + ".normalizedTangent", up, jp + ".position", jt,
                     "{}_frame{}".format(name, j))
        joints.append(jt)

    return {"grp": grp, "cv_ctrls": cv_ctrls, "twist_ctrl": twist_ctrls,
            "joints": joints, "curve": curve_shape, "up_curve": up_curve,
            "nseg": nseg, "out_ctrl": out_ctrl, "in_ctrl": in_ctrl,
            "out_buf": out_buf, "in_buf": in_buf,
            "rest_off_out": rest_off_out, "rest_off_in": rest_off_in}


def adapt_shared_controls(grp, cv_ctrls, twist_ctrls, o_ctrls, i_ctrls,
                          o_bufs, i_bufs, o_rests, i_rests):
    """Map the production control rig (builder.mkTwistSplineControllers) into the
    dict build_native_spline(controls=...) expects. The per-segment tangent lists
    are padded to length n (out at i=0..n-2, in at i=1..n-1), and the rest buffers
    are decomposed to world-space point plugs (the Auto=0 reference).
    """
    return {
        "grp": grp,
        "cv_ctrls": list(cv_ctrls),
        "twist_ctrl": list(twist_ctrls),
        "out_ctrl": list(o_ctrls) + [None],
        "in_ctrl": [None] + list(i_ctrls),
        "out_buf": list(o_bufs) + [None],
        "in_buf": [None] + list(i_bufs),
        "rest_out": [_decompose(o) for o in o_rests] + [None],
        "rest_in": [None] + [_decompose(o) for o in i_rests],
    }


def verify_frames(rig, spread=3.0):
    """Compare joint position + (twisted) orientation to the validated spec.

    Reads the live Twist/UseTwist off the CV controls, evaluates native_ref with
    every CV pinned (the native rig's regime), and matches each joint to the spec
    by position-inversion (param-space agnostic)."""
    import maya.api.OpenMaya as om
    cv_pos = [cmds.xform(c, q=True, ws=True, t=True) for c in rig["cv_ctrls"]]
    n = len(rig["cv_ctrls"])
    # raw twist (rotateX) as values, UseTwist as the lock weights -> native_ref
    # floats unpinned CVs between twist pins, matching the rig's twist solve.
    twist_vals = [math.radians(cmds.getAttr(tc + ".rotateX")) for tc in rig["twist_ctrl"]]
    twist_locks = [cmds.getAttr(tc + ".UseTwist") for tc in rig["twist_ctrl"]]

    cv_quats, orient_locks = [], []
    for c in rig["cv_ctrls"]:
        q = om.MTransformationMatrix(
            om.MMatrix(cmds.xform(c, q=True, ws=True, matrix=True))).rotation(asQuaternion=True)
        cv_quats.append([q.w, q.x, q.y, q.z])
        orient_locks.append(1.0 if cmds.getAttr(c + ".UseOrient") >= 0.5 else 0.0)

    # Use the curve's LIVE bezier handles so the reference matches any tangent
    # edits (Weight / Smooth / Auto / manual moves), not just the default shape.
    nseg = rig["nseg"]
    in_tans = [None] * n
    out_tans = [None] * n
    for k in range(nseg):
        out_tans[k] = cmds.pointPosition("{}.cv[{}]".format(rig["curve"], 3 * k + 1), world=True)
        in_tans[k + 1] = cmds.pointPosition("{}.cv[{}]".format(rig["curve"], 3 * k + 2), world=True)
    tangents = (in_tans, out_tans)

    # native_ref param grid (covers the curve), then position-match each joint.
    # The reference RMF samples uniformly in arc length, so it needs a high sample
    # count to resolve short, sharply-turning regions (e.g. a hairpin from a
    # dragged tangent) that the rig's per-segment curve-param sampling handles
    # natively -- otherwise the *reference* under-resolves and falsely flags the rig.
    from ..core import make_spline
    rng = make_spline(cv_pos, spread=spread, tangents=tangents).param_range
    grid = [rng[0] + (rng[1] - rng[0]) * i / 2000 for i in range(2001)]
    ref = native_frames(cv_pos, grid, cv_quats=cv_quats, twist_vals=twist_vals,
                        twist_locks=twist_locks, orient_locks=orient_locks, spread=spread,
                        tangents=tangents, samples_per_seg=256)

    max_pos = max_frame = 0.0
    for j in rig["joints"]:
        jp = cmds.xform(j, q=True, ws=True, t=True)
        gi = min(range(len(ref)), key=lambda k: vm.length(vm.sub(ref[k]["pos"], jp)))
        f = ref[gi]
        max_pos = max(max_pos, vm.length(vm.sub(jp, f["pos"])))
        wm = cmds.xform(j, q=True, ws=True, matrix=True)
        jy = vm.normalized([wm[4], wm[5], wm[6]])  # joint world Y = its normal
        d = max(-1.0, min(1.0, vm.dot(jy, f["normal"])))
        max_frame = max(max_frame, math.degrees(math.acos(d)))
    ok = max_pos < 1e-2 and max_frame < 1.0
    print("native frames vs spec | max dPos={:.3e} | max dFrame={:.3f} deg | {}".format(
        max_pos, max_frame, "OK" if ok else "REVIEW"))
    return max_pos, max_frame


def verify_tangents(rig):
    """Compare the live AUTO tangent (the buffer the control rides) to the faithful
    tangent reference (multi_tangent_handles), reading the current Weight/Smooth/
    Auto. The reference's manual input is the rig's rest offset, so this validates
    the auto<->rest blend; the control's own offset (which always bends the spline)
    is exercised by the C++ parity sweep instead."""
    n = len(rig["cv_ctrls"])
    cvs = [cmds.xform(c, q=True, ws=True, t=True) for c in rig["cv_ctrls"]]
    in_w, out_w = [1.0] * n, [1.0] * n
    in_s, out_s = [1.0] * n, [1.0] * n
    in_a, out_a = [1.0] * n, [1.0] * n
    in_user = [list(v) if v else [0.0, 0.0, 0.0] for v in rig["rest_off_in"]]
    out_user = [list(v) if v else [0.0, 0.0, 0.0] for v in rig["rest_off_out"]]
    for i in range(n):
        oc, ic = rig["out_ctrl"][i], rig["in_ctrl"][i]
        if oc:
            out_w[i], out_s[i], out_a[i] = (cmds.getAttr(oc + ".Weight"),
                                            cmds.getAttr(oc + ".Smooth"),
                                            cmds.getAttr(oc + ".Auto"))
        if ic:
            in_w[i], in_s[i], in_a[i] = (cmds.getAttr(ic + ".Weight"),
                                         cmds.getAttr(ic + ".Smooth"),
                                         cmds.getAttr(ic + ".Auto"))
    inh, outh = multi_tangent_handles(cvs, in_w, out_w, in_s, out_s,
                                      in_a, out_a, in_user, out_user)
    nseg = rig["nseg"]
    maxd = 0.0
    for k in range(nseg):
        op = cmds.xform(rig["out_buf"][k], q=True, ws=True, t=True)
        ip = cmds.xform(rig["in_buf"][k + 1], q=True, ws=True, t=True)
        maxd = max(maxd, vm.length(vm.sub(op, outh[k])))
        maxd = max(maxd, vm.length(vm.sub(ip, inh[k + 1])))
    ok = maxd < 1e-4
    print("native tangents vs reference | max dHandle={:.3e} | {}".format(
        maxd, "OK" if ok else "REVIEW"))
    return maxd


# ---- cosmetic finalize (animator-facing; math/parity untouched) -----------

def _cube_pts(s):
    return [[-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s], [-s, -s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s], [-s, -s, s],
            [s, -s, s], [s, -s, -s], [s, s, -s], [s, s, s], [-s, s, s], [-s, s, -s]]


def _ctrl_shape(transform, kind, size, color):
    """Swap a control's locator shape for a colored NURBS curve shape."""
    old = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
    if kind == "cube":
        tmp = cmds.curve(degree=1, point=_cube_pts(size))
    else:  # circleX / circleY / circleZ
        nrm = {"circleX": [1, 0, 0], "circleY": [0, 1, 0],
               "circleZ": [0, 0, 1]}.get(kind, [1, 0, 0])
        tmp = cmds.circle(normal=nrm, radius=size, constructionHistory=False)[0]
    for s in cmds.listRelatives(tmp, shapes=True, fullPath=True) or []:
        cmds.setAttr(s + ".overrideEnabled", 1)
        cmds.setAttr(s + ".overrideColor", color)
        cmds.parent(s, transform, shape=True, relative=True)
    cmds.delete(tmp)
    for s in old:
        cmds.delete(s)


def _lock_hide(node, attrs):
    for a in attrs:
        try:
            cmds.setAttr(node + "." + a, lock=True, keyable=False, channelBox=False)
        except Exception:
            pass


def finalize_rig(rig):
    """Make a built rig animator-friendly: colored NURBS control shapes, internal
    guts hidden (upCurve + arcLengthDimension nodes), non-animatable channels
    locked/hidden, and a selection set of the controls. Purely cosmetic -- the
    node math and C++ parity are unaffected. Call after build_native_spline.
    """
    grp = rig["grp"]
    name = grp.rsplit("|", 1)[-1]
    if name.endswith("_grp"):
        name = name[:-4]

    # hidden 'guts' group for the internal display geometry
    guts = cmds.createNode("transform", name=name + "_guts", parent=grp)
    cmds.setAttr(guts + ".visibility", 0)
    upT = cmds.listRelatives(rig["up_curve"], parent=True, fullPath=True) or []
    for t in upT:
        cmds.parent(t, guts)
    for s in set(cmds.listConnections(rig["curve"] + ".worldSpace[0]",
                                      type="arcLengthDimension") or []):
        for t in cmds.listRelatives(s, parent=True, fullPath=True) or []:
            cmds.parent(t, guts)

    # colored shapes: CV=yellow circle, twist=red dial (around X), tangents=cubes
    for c in rig["cv_ctrls"]:
        _ctrl_shape(c, "circleX", 1.0, 17)
    for c in rig["twist_ctrl"]:
        _ctrl_shape(c, "circleX", 0.6, 13)
    for c in rig["out_ctrl"]:
        if c:
            _ctrl_shape(c, "cube", 0.28, 6)
    for c in rig["in_ctrl"]:
        if c:
            _ctrl_shape(c, "cube", 0.28, 18)

    # channel cleanup (keep only the animatable channels + custom attrs)
    for c in rig["cv_ctrls"]:
        _lock_hide(c, ["sx", "sy", "sz", "v"])
    for c in rig["twist_ctrl"]:
        _lock_hide(c, ["tx", "ty", "tz", "ry", "rz", "sx", "sy", "sz", "v"])
    for c in rig["out_ctrl"] + rig["in_ctrl"]:
        if c:
            _lock_hide(c, ["rx", "ry", "rz", "sx", "sy", "sz", "v"])
    for j in rig["joints"]:
        _lock_hide(j, ["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz", "v"])

    anim = (rig["cv_ctrls"] + rig["twist_ctrl"]
            + [c for c in rig["out_ctrl"] + rig["in_ctrl"] if c])
    rig["set"] = cmds.sets(anim, name=name + "_controls")
    rig["guts"] = guts
    return rig
