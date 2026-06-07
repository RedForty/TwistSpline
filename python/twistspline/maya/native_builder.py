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


# ---- control attributes ---------------------------------------------------

def _add_cv_attrs(ctrl):
    for nm, default in (("Twist", 0.0), ("UseTwist", 0.0), ("UseOrient", 0.0)):
        cmds.addAttr(ctrl, longName=nm, attributeType="double", defaultValue=default,
                     keyable=True)


# ---- Stage 1 build --------------------------------------------------------

def build_native_spline(cv_positions, num_joints, spread=3.0, name="nativeTS",
                        samples_per_interval=8):
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
        cmds.parent(c, grp)
        cmds.xform(c, worldSpace=True, translation=cvs[i])
        _add_cv_attrs(c)
        cv_ctrls.append(c)
    # Default pin pattern (matches the real rig): twist@CV0, orient@first+last.
    cmds.setAttr(cv_ctrls[0] + ".UseTwist", 1.0)
    cmds.setAttr(cv_ctrls[0] + ".UseOrient", 1.0)
    cmds.setAttr(cv_ctrls[-1] + ".UseOrient", 1.0)

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

    # ---- Stage 2: one parallel-transport chain, sampled by curve parameter.
    # Joints sit AT chain nodes (param p_j); between consecutive joints we insert
    # `samples_per_interval-1` fill samples for transport accuracy. All params are
    # fixed, so the chain is static topology.
    sample_params, joint_idx = [], []
    for j in range(num_joints):
        joint_idx.append(len(sample_params))
        sample_params.append(j / (num_joints - 1.0) * nseg if num_joints > 1 else 0.0)
        if j < num_joints - 1:
            a = j / (num_joints - 1.0) * nseg
            b = (j + 1) / (num_joints - 1.0) * nseg
            for s in range(1, samples_per_interval):
                sample_params.append(a + (b - a) * s / samples_per_interval)

    pocis = [_poci(curve_shape, p, "{}_poci{}".format(name, i))
             for i, p in enumerate(sample_params)]
    tans = [pc + ".normalizedTangent" for pc in pocis]

    # anchor up = CV0 control Y, reprojected perpendicular to the start tangent
    ups = [None] * len(pocis)
    ups[0] = _reproject(_world_y(cv_ctrls[0], name + "_anchorY"), tans[0], name + "_anchor")
    for i in range(1, len(pocis)):
        ups[i] = _transport(ups[i - 1], tans[i - 1], tans[i], "{}_tr{}".format(name, i))

    joints = []
    for j in range(num_joints):
        idx = joint_idx[j]
        jt = cmds.createNode("joint", name="{}_jnt{}".format(name, j), parent=grp)
        _build_frame(tans[idx], ups[idx], pocis[idx] + ".position", jt,
                     "{}_frame{}".format(name, j))
        joints.append(jt)

    return {"grp": grp, "cv_ctrls": cv_ctrls, "joints": joints,
            "curve": curve_shape, "nseg": nseg,
            "sample_params": [sample_params[i] for i in joint_idx]}


def verify_frames(rig, spread=3.0):
    """Compare joint position + orientation to the kernel at the joints' actual
    points on the curve (position-inversion, so it's param-space agnostic)."""
    from ..core import make_spline
    cv_pos = [cmds.xform(c, q=True, ws=True, t=True) for c in rig["cv_ctrls"]]
    ker = make_spline(cv_pos, spread=spread)
    lo, hi = ker.param_range
    grid = [lo + (hi - lo) * i / 2000 for i in range(2001)]
    gpos = [ker.matrix_at_param(g, twisted=False).tran for g in grid]

    max_pos = max_frame = 0.0
    for j in rig["joints"]:
        jp = cmds.xform(j, q=True, ws=True, t=True)
        # nearest kernel param to this joint's world position
        gi = min(range(len(grid)), key=lambda k: vm.length(vm.sub(gpos[k], jp)))
        kf = ker.matrix_at_param(grid[gi], twisted=True)
        max_pos = max(max_pos, vm.length(vm.sub(jp, kf.tran)))
        # joint's world Y axis (its normal) from its worldMatrix
        wm = cmds.xform(j, q=True, ws=True, matrix=True)
        jy = vm.normalized([wm[4], wm[5], wm[6]])
        d = max(-1.0, min(1.0, vm.dot(jy, kf.norm)))
        max_frame = max(max_frame, math.degrees(math.acos(d)))
    ok = max_pos < 1e-2 and max_frame < 1.0
    print("native frames vs kernel | max dPos={:.3e} | max dFrame={:.3f} deg | {}".format(
        max_pos, max_frame, "OK" if ok else "REVIEW"))
    return max_pos, max_frame
