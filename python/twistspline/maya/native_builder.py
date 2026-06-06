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


# ---- control attributes ---------------------------------------------------

def _add_cv_attrs(ctrl):
    for nm, default in (("Twist", 0.0), ("UseTwist", 0.0), ("UseOrient", 0.0)):
        cmds.addAttr(ctrl, longName=nm, attributeType="double", defaultValue=default,
                     keyable=True)


# ---- Stage 1 build --------------------------------------------------------

def build_native_spline(cv_positions, num_joints, spread=3.0, name="nativeTS"):
    """Stage 1: live curve from CV controls + motionPath joints + control attrs.

    Returns a dict of node names (cv controls, joints, curve, group).
    """
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

    # Joints riding the curve by arc-length fraction (motionPath, Stage 1 = pos only).
    joints = []
    for i in range(num_joints):
        frac = i / (num_joints - 1.0) if num_joints > 1 else 0.0
        j = cmds.createNode("joint", name="{}_jnt{}".format(name, i), parent=grp)
        mp = cmds.createNode("motionPath", name="{}_mp{}".format(name, i))
        cmds.connectAttr(curve_shape + ".worldSpace[0]", mp + ".geometryPath")
        cmds.setAttr(mp + ".fractionMode", 1)
        cmds.setAttr(mp + ".uValue", frac)
        cmds.connectAttr(mp + ".allCoordinates", j + ".translate")
        joints.append(j)

    return {"grp": grp, "cv_ctrls": cv_ctrls, "joints": joints,
            "curve": curve_shape, "nseg": nseg}


def verify_positions(rig, spread=3.0):
    """Compare Stage-1 joint positions to the validated spec (native_ref)."""
    cv_pos = [cmds.xform(c, q=True, ws=True, t=True) for c in rig["cv_ctrls"]]
    m = len(rig["joints"])
    fracs = [i / (m - 1.0) if m > 1 else 0.0 for i in range(m)]
    params = params_at_fractions(cv_pos, fracs, spread=spread)
    ref = native_frames(cv_pos, params, spread=spread)

    max_d = 0.0
    for j, fr in zip(rig["joints"], ref):
        jp = cmds.xform(j, q=True, ws=True, t=True)
        max_d = max(max_d, vm.length(vm.sub(jp, fr["pos"])))
    print("native positions vs spec | max dPos={:.4e} | {}".format(
        max_d, "OK" if max_d < 1e-4 else "REVIEW"))
    return max_d
