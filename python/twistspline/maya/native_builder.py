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
    cmds.connectAttr(ab + ".axis", aaq + ".inputAxis")
    cmds.connectAttr(ab + ".angle", aaq + ".inputAngle")
    cm = cmds.createNode("composeMatrix", name=name + "_cm")
    cmds.setAttr(cm + ".useEulerRotation", 0)
    cmds.connectAttr(aaq + ".outputQuat", cm + ".inputQuat")
    pmm = cmds.createNode("vectorProduct", name=name + "_rot")
    cmds.setAttr(pmm + ".operation", 3)  # vector-matrix product
    cmds.connectAttr(up_prev, pmm + ".input1")
    cmds.connectAttr(cm + ".outputMatrix", pmm + ".matrix")
    return _reproject(pmm + ".output", t_cur, name + "_rp")


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
    cmds.addAttr(ctrl, longName="Twist", attributeType="doubleAngle",
                 defaultValue=0.0, keyable=True)
    cmds.addAttr(ctrl, longName="UseTwist", attributeType="double",
                 defaultValue=1.0, min=0.0, max=1.0, keyable=True)
    cmds.addAttr(ctrl, longName="UseOrient", attributeType="double",
                 defaultValue=0.0, min=0.0, max=1.0, keyable=True)
    cmds.addAttr(ctrl, longName="Pin", attributeType="double",
                 defaultValue=0.0, min=0.0, max=1.0, keyable=True)
    cmds.addAttr(ctrl, longName="PinParam", attributeType="double",
                 defaultValue=0.0, keyable=True)


# ---- Stage 1 build --------------------------------------------------------

def build_native_spline(cv_positions, num_joints, spread=3.0, name="nativeTS",
                        samples_per_interval=16, pins=None):
    """Live curve from CV controls + RMF-oriented joints + per-CV control attrs.

    Stage 1+2: position via the live degree-3 curve, orientation via a
    parallel-transport RMF chain. Returns a dict of node names.
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
    out_tans, in_tans = _catmull_tangents(cvs)

    grp = cmds.createNode("transform", name=name + "_grp")
    curve_tfm = cmds.createNode("transform", name=name + "_curve", parent=grp)

    # CV control transforms with the per-CV attributes.
    cv_ctrls = []
    for i in range(n):
        c = cmds.spaceLocator(name="{}_cv{}".format(name, i))[0]
        # full DAG path so a same-named rebuild in one scene stays unambiguous
        c = cmds.ls(cmds.parent(c, grp)[0], long=True)[0]
        cmds.xform(c, worldSpace=True, translation=cvs[i])
        _add_cv_attrs(c)
        cv_ctrls.append(c)
    # Default pin pattern (matches the real rig): twist@CV0, orient@first+last.
    cmds.setAttr(cv_ctrls[0] + ".UseTwist", 1.0)
    cmds.setAttr(cv_ctrls[0] + ".UseOrient", 1.0)
    cmds.setAttr(cv_ctrls[-1] + ".UseOrient", 1.0)
    # Position pins (param-map anchors) requested at build time.
    for k in (pins or []):
        cmds.setAttr(cv_ctrls[k] + ".Pin", 1.0)

    cv_pos = [_decompose(c) for c in cv_ctrls]

    # Auto bezier tangents (Catmull-Rom) computed live by nodes.
    out_plug = [None] * n
    in_plug = [None] * n
    for k in range(n):
        cur = cv_pos[k]
        if k == 0:
            m = _sub(cv_pos[1], cur, "{}_m{}".format(name, k))
        elif k == n - 1:
            m = _sub(cur, cv_pos[k - 1], "{}_m{}".format(name, k))
        else:
            m = _scale(_sub(cv_pos[k + 1], cv_pos[k - 1], "{}_mr{}".format(name, k)),
                       0.5, "{}_m{}".format(name, k))
        m3 = _scale(m, 1.0 / 3.0, "{}_m3_{}".format(name, k))
        if k < n - 1:
            out_plug[k] = _add(cur, m3, "{}_out{}".format(name, k))
        if k > 0:
            in_plug[k] = _sub(cur, m3, "{}_in{}".format(name, k))

    # Degree-3 bezier-form curve: control points [cv0, out0, in1, cv1, out1, ...].
    init_pts = []
    for k in range(nseg):
        init_pts.extend([cvs[k], out_tans[k], in_tans[k + 1]])
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

    # Drive every control point from the live CV / tangent plugs.
    for k in range(nseg):
        cmds.connectAttr(cv_pos[k], "{}.controlPoints[{}]".format(curve_shape, 3 * k))
        cmds.connectAttr(out_plug[k], "{}.controlPoints[{}]".format(curve_shape, 3 * k + 1))
        cmds.connectAttr(in_plug[k + 1], "{}.controlPoints[{}]".format(curve_shape, 3 * k + 2))
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

    # anchor up = CV0 control Y, reprojected perpendicular to the start tangent
    ups = [None] * len(pocis)
    ups[0] = _reproject(_world_y(cv_ctrls[0], name + "_anchorY"), tans[0], name + "_anchor")
    for i in range(1, len(pocis)):
        ups[i] = _transport(ups[i - 1], tans[i - 1], tans[i], "{}_tr{}".format(name, i))

    # Stage 3: per-CV twist. Every CV is a twist knot (value = Twist * UseTwist),
    # interpolated piecewise-linearly by *live* arc length between consecutive CVs.
    arc_cv = [_alen(curve_shape, float(k), "{}_alenCV{}".format(name, k))
              for k in range(nseg + 1)]
    twist_val = [_mul1(cv_ctrls[k] + ".Twist", cv_ctrls[k] + ".UseTwist",
                       "{}_tval{}".format(name, k)) for k in range(n)]

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
    # holds its PinParam; an UNPINNED CV floats to its arc-length position --
    # interpolated by *live* arc length between the bracketing pins. (This is the
    # exact reduction of solveParamMatrix under 0/1 pins.) Endpoints are always
    # anchors so the param range is well defined. Pin SET is read at build time;
    # PinParam values and arc lengths (hence the stretch) are fully live.
    pin_set = sorted(set([0, n - 1]
                         + [k for k in range(n) if cmds.getAttr(cv_ctrls[k] + ".Pin") >= 0.5]))
    remap = [None] * n
    for k in pin_set:
        remap[k] = cv_ctrls[k] + ".PinParam"
    for k in range(n):
        if remap[k] is not None:
            continue
        lo = max(a for a in pin_set if a < k)
        hi = min(a for a in pin_set if a > k)
        frac = _mul1(_sub1(arc_cv[k], arc_cv[lo], "{}_rmn{}".format(name, k)),
                     _sub1(arc_cv[hi], arc_cv[lo], "{}_rmd{}".format(name, k)),
                     "{}_rmf{}".format(name, k), divide=True)
        remap[k] = _lerp1(cv_ctrls[lo] + ".PinParam", cv_ctrls[hi] + ".PinParam",
                          frac, "{}_rmp{}".format(name, k))

    # Rest remap (numeric, evaluated off the live curve at rest) -- used ONLY to
    # assign each joint the segment its rider param falls in. Stays valid as long
    # as pins keep that param inside the segment (true for moderate pinning).
    arc_rest = [cmds.getAttr(a) for a in arc_cv]
    pin_par = [cmds.getAttr(cv_ctrls[k] + ".PinParam") for k in range(n)]
    rest_remap = list(pin_par)
    for k in range(n):
        if k in pin_set:
            continue
        lo = max(a for a in pin_set if a < k)
        hi = min(a for a in pin_set if a > k)
        fr = (arc_rest[k] - arc_rest[lo]) / (arc_rest[hi] - arc_rest[lo])
        rest_remap[k] = pin_par[lo] + fr * (pin_par[hi] - pin_par[lo])
    pmin, pmax = rest_remap[0], rest_remap[-1]

    # ---- joints: rider param -> live curve param via the remap, then sample both
    # curves there. u_live = seg + (t_j - remap[seg]) / (remap[seg+1] - remap[seg]).
    joints = []
    for j in range(num_joints):
        t_j = pmin + (pmax - pmin) * j / (num_joints - 1.0) if num_joints > 1 else pmin
        seg = min(max([k for k in range(nseg) if rest_remap[k] <= t_j + 1e-9],
                      default=0), nseg - 1)
        local = _mul1(_sub_cp(t_j, remap[seg], "{}_jln{}".format(name, j)),
                      _sub1(remap[seg + 1], remap[seg], "{}_jld{}".format(name, j)),
                      "{}_jlf{}".format(name, j), divide=True)
        u_live = _add_pc(local, float(seg), "{}_ju{}".format(name, j))
        up_par = _scale1(u_live, float(samples_per_interval), "{}_jus{}".format(name, j))
        jp = _poci_live(curve_shape, u_live, "{}_jp{}".format(name, j))
        jup = _poci_live(up_curve, up_par, "{}_jup{}".format(name, j))
        up = _reproject(jup + ".position", jp + ".normalizedTangent", "{}_jrp{}".format(name, j))
        jt = cmds.createNode("joint", name="{}_jnt{}".format(name, j), parent=grp)
        _build_frame(jp + ".normalizedTangent", up, jp + ".position", jt,
                     "{}_frame{}".format(name, j))
        joints.append(jt)

    return {"grp": grp, "cv_ctrls": cv_ctrls, "joints": joints,
            "curve": curve_shape, "up_curve": up_curve, "nseg": nseg}


def verify_frames(rig, spread=3.0):
    """Compare joint position + (twisted) orientation to the validated spec.

    Reads the live Twist/UseTwist off the CV controls, evaluates native_ref with
    every CV pinned (the native rig's regime), and matches each joint to the spec
    by position-inversion (param-space agnostic)."""
    import maya.api.OpenMaya as om
    cv_pos = [cmds.xform(c, q=True, ws=True, t=True) for c in rig["cv_ctrls"]]
    n = len(rig["cv_ctrls"])
    twist_vals = [math.radians(cmds.getAttr(c + ".Twist"))  # doubleAngle -> degrees
                  * cmds.getAttr(c + ".UseTwist") for c in rig["cv_ctrls"]]
    twist_locks = [1.0] * n

    cv_quats, orient_locks = [], []
    for c in rig["cv_ctrls"]:
        q = om.MTransformationMatrix(
            om.MMatrix(cmds.xform(c, q=True, ws=True, matrix=True))).rotation(asQuaternion=True)
        cv_quats.append([q.w, q.x, q.y, q.z])
        orient_locks.append(1.0 if cmds.getAttr(c + ".UseOrient") >= 0.5 else 0.0)

    # native_ref param grid (covers the curve), then position-match each joint.
    from ..core import make_spline
    rng = make_spline(cv_pos, spread=spread).param_range
    grid = [rng[0] + (rng[1] - rng[0]) * i / 2000 for i in range(2001)]
    ref = native_frames(cv_pos, grid, cv_quats=cv_quats, twist_vals=twist_vals,
                        twist_locks=twist_locks, orient_locks=orient_locks, spread=spread)

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
