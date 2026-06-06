"""
Helpers to create and verify a ``pyTwistSpline`` node in-scene (Phase 1).

``create_py_spline`` makes the node plus a set of control locators (CVs and
auto bezier tangents) wired into its ``vertexData`` -- the same data flow the
C++ rig uses -- so you get a spline drawing in the viewport from controls you
can move.

``read_node_spline`` pulls the computed :class:`TwistSpline` back off the
output plug, and ``verify_against_kernel`` confirms the node produces exactly
what calling the kernel directly would, proving the attribute-reading + MPxData
round-trip is correct.
"""

import maya.cmds as cmds
import maya.api.OpenMaya as om

from ..core import make_spline, _catmull_tangents
from .. import _vmath as vm


def _make_controls(cvs, name):
    """Create CV + auto-tangent locators for a control hull.

    Returns (cv_locs, out_locs, in_locs); in_locs[i] is the in-tangent toward
    cv[i+1], matching the C++ builder's connection layout.
    """
    n = len(cvs)
    out_tans, in_tans = _catmull_tangents(cvs)

    def _loc(pos, nm):
        loc = cmds.spaceLocator(name=nm)[0]
        cmds.xform(loc, worldSpace=True, translation=pos)
        return loc

    cv_locs = [_loc(cvs[i], "{}_cv{}".format(name, i)) for i in range(n)]
    out_locs = [_loc(out_tans[i], "{}_out{}".format(name, i)) for i in range(n - 1)]
    in_locs = [_loc(in_tans[i + 1], "{}_in{}".format(name, i + 1)) for i in range(n - 1)]
    return cv_locs, out_locs, in_locs


def _wire_spline(node_type, shape_name, cv_locs, out_locs, in_locs, spread):
    """Create a spline node of ``node_type`` and wire the controls into it.

    Works for both ``pyTwistSpline`` and the C++ ``twistSpline`` (identical
    attribute names). Returns the shape node.
    """
    shape = cmds.createNode(node_type, name=shape_name)
    n = len(cv_locs)
    for i in range(n):
        vd = "{}.vertexData[{}]".format(shape, i)
        cmds.connectAttr(cv_locs[i] + ".worldMatrix[0]", vd + ".controlVertex")
        cmds.setAttr(vd + ".paramValue", i * spread)
        if i < n - 1:
            cmds.connectAttr(out_locs[i] + ".worldMatrix[0]", vd + ".outTangent")
        if i > 0:
            cmds.connectAttr(in_locs[i - 1] + ".worldMatrix[0]", vd + ".inTangent")
    cmds.setAttr(shape + ".vertexData[0].paramWeight", 1.0)
    cmds.setAttr(shape + ".vertexData[0].twistWeight", 1.0)
    return shape


def create_py_spline(cv_positions, spread=3.0, name="pyTwistSpline"):
    """Create a pyTwistSpline node + CV/tangent locators, wired up.

    Returns (shape, cv_locators, tangent_locators).
    """
    if not cmds.pluginInfo("pyTwistSplinePlugin", q=True, loaded=True):
        raise RuntimeError("Load the plugin first: "
                           "cmds.loadPlugin('.../python/pyTwistSplinePlugin.py')")

    cvs = [list(p) for p in cv_positions]
    cv_locs, out_locs, in_locs = _make_controls(cvs, name)
    shape = _wire_spline("pyTwistSpline", name + "Shape", cv_locs, out_locs, in_locs, spread)
    cmds.setAttr(shape + ".debugDisplay", True)
    return shape, cv_locs, out_locs + in_locs


def read_node_spline(shape):
    """Pull the computed twistspline.core.TwistSpline off the output plug."""
    sel = om.MSelectionList()
    sel.add(shape)
    node = sel.getDependNode(0)
    plug = om.MFnDependencyNode(node).findPlug("outputSpline", False)
    obj = plug.asMObject()
    return om.MFnPluginData(obj).data().spline()


def verify_against_kernel(shape, cv_positions, samples=50):
    """Compare the node's output spline to a kernel-built one (same inputs).

    Returns (max_pos_err, max_norm_err). Both should be ~0 -- this checks the
    Maya plumbing, since the underlying math is identical.
    """
    node_spline = read_node_spline(shape)
    ref = make_spline(cv_positions)

    lo, hi = ref.param_range
    max_pos = max_norm = 0.0
    for i in range(samples):
        t = lo + (hi - lo) * i / (samples - 1)
        a = node_spline.matrix_at_param(t)
        b = ref.matrix_at_param(t)
        max_pos = max(max_pos, vm.length(vm.sub(a.tran, b.tran)))
        max_norm = max(max_norm, vm.length(vm.sub(a.norm, b.norm)))

    ok = max_pos < 1e-9 and max_norm < 1e-9
    print("node vs kernel | max dPos={:.3e} | max dNorm={:.3e} | {}".format(
        max_pos, max_norm, "OK" if ok else "MISMATCH"))
    return max_pos, max_norm


def verify_twist_read(shape, cv_positions, cv_index=2, degrees=90.0, samples=50):
    """Confirm the node reads ``twistValue`` (a kAngle attr) in radians.

    Sets a twist on one CV, then reads the plug back as *true internal radians*
    via the API (sidestepping all UI-unit ambiguity), builds a kernel reference
    spline with that exact radian twist, and compares. If the node's compute
    read the angle correctly the two match; if it mistook degrees for radians
    they diverge wildly.
    """
    import math

    attr = "{}.vertexData[{}]".format(shape, cv_index)
    cmds.setAttr(attr + ".twistValue", degrees)
    cmds.setAttr(attr + ".twistWeight", 1.0)

    # The exact radian value the node's compute will see off the handle.
    sel = om.MSelectionList()
    sel.add(attr + ".twistValue")
    true_rad = sel.getPlug(0).asMAngle().asRadians()

    node_spline = read_node_spline(shape)

    n = len(cv_positions)
    user_twists = [0.0] * n
    user_twists[cv_index] = true_rad
    twist_locks = [0.0] * n
    twist_locks[0] = 1.0
    twist_locks[cv_index] = 1.0
    ref = make_spline(cv_positions, user_twists=user_twists, twist_locks=twist_locks)

    lo, hi = ref.param_range
    max_pos = max_norm = 0.0
    for i in range(samples):
        t = lo + (hi - lo) * i / (samples - 1)
        a = node_spline.matrix_at_param(t)
        b = ref.matrix_at_param(t)
        max_pos = max(max_pos, vm.length(vm.sub(a.tran, b.tran)))
        max_norm = max(max_norm, vm.length(vm.sub(a.norm, b.norm)))

    ok = max_pos < 1e-9 and max_norm < 1e-9
    print("twist read | set {:.1f}deg -> {:.6f}rad | max dNorm={:.3e} | {}".format(
        degrees, true_rad, max_norm,
        "OK (radians)" if ok else "MISMATCH (node likely read degrees)"))
    return max_pos, max_norm


# ---------------------------------------------------------------------------
# Phase 2: parity against the C++ riderConstraint
# ---------------------------------------------------------------------------

# Settable single-valued attributes shared by both rider node types. Maps a
# friendly key to the attribute name.
_RIDER_GLOBALS = {
    "rotateOrder": "rotateOrder", "globalOffset": "globalOffset",
    "globalSpread": "globalSpread", "scaleCompensation": "scaleCompensation",
    "useCycle": "useCycle", "normalize": "normalize", "normValue": "normValue",
    "useGlobalMin": "useGlobalMin", "minGlobalParam": "minGlobalParam",
    "useGlobalMax": "useGlobalMax", "maxGlobalParam": "maxGlobalParam",
}


def _make_rider(node_type, name, spline_specs, params, rider_globals, parent_inv):
    """Create a rider of ``node_type`` fed by one or more weighted splines.

    spline_specs : list of (spline_shape, weight). params : list of dicts.
    parent_inv : transform whose worldInverseMatrix feeds every param's
    parentInverseMatrix (exercises that path).
    """
    rider = cmds.createNode(node_type, name=name)
    for si, (spline_shape, weight) in enumerate(spline_specs):
        cmds.connectAttr(spline_shape + ".outputSpline",
                         "{}.inputSplines[{}].spline".format(rider, si))
        cmds.connectAttr(spline_shape + ".splineLength",
                         "{}.inputSplines[{}].splineLength".format(rider, si))
        cmds.setAttr("{}.inputSplines[{}].weight".format(rider, si), weight)

    for k, v in (rider_globals or {}).items():
        cmds.setAttr("{}.{}".format(rider, _RIDER_GLOBALS[k]), v)

    for i, pspec in enumerate(params):
        base = "{}.params[{}]".format(rider, i)
        cmds.setAttr(base + ".param", pspec.get("param", 0.0))
        if "useMin" in pspec:
            cmds.setAttr(base + ".useMin", pspec["useMin"])
            cmds.setAttr(base + ".minParam", pspec.get("minParam", 0.0))
        if "useMax" in pspec:
            cmds.setAttr(base + ".useMax", pspec["useMax"])
            cmds.setAttr(base + ".maxParam", pspec.get("maxParam", 1.0))
        if parent_inv is not None:
            cmds.connectAttr(parent_inv + ".worldInverseMatrix[0]",
                             base + ".parentInverseMatrix")
    return rider


def create_parity_rig(cv_positions, params, spread=3.0, rider_globals=None,
                      use_parent=True, name="parity"):
    """Build parallel py / C++ spline+rider chains from a *single* control hull.

    Both the Python and C++ riders read geometrically-identical splines (proven
    equal in Phase 0/1) driven by the same locators, so any output difference is
    a rider-port bug. Returns a dict of node names.
    """
    if not cmds.pluginInfo("TwistSpline", q=True, loaded=True):
        raise RuntimeError("Load the C++ plugin first (cmds.loadPlugin('TwistSpline')).")
    if not cmds.pluginInfo("pyTwistSplinePlugin", q=True, loaded=True):
        raise RuntimeError("Load the Python plugin first.")

    cvs = [list(p) for p in cv_positions]
    cv_locs, out_locs, in_locs = _make_controls(cvs, name + "_ctl")

    py_spline = _wire_spline("pyTwistSpline", name + "_pySplineShape",
                             cv_locs, out_locs, in_locs, spread)
    cpp_spline = _wire_spline("twistSpline", name + "_cppSplineShape",
                              cv_locs, out_locs, in_locs, spread)

    parent = None
    if use_parent:
        parent = cmds.createNode("transform", name=name + "_rigParent")
        cmds.setAttr(parent + ".translate", 1.5, -2.0, 0.75)
        cmds.setAttr(parent + ".rotate", 20.0, -35.0, 12.0)
        cmds.setAttr(parent + ".scale", 1.0, 1.0, 1.0)

    py_rider = _make_rider("pyRiderConstraint", name + "_pyRider",
                           [(py_spline, 1.0)], params, rider_globals, parent)
    cpp_rider = _make_rider("riderConstraint", name + "_cppRider",
                            [(cpp_spline, 1.0)], params, rider_globals, parent)

    return {
        "py_spline": py_spline, "cpp_spline": cpp_spline,
        "py_rider": py_rider, "cpp_rider": cpp_rider,
        "parent": parent, "cv_locs": cv_locs, "nparams": len(params),
    }


def create_multi_parity_rig(cv_hulls, params, weights, spread=3.0,
                            rider_globals=None, use_parent=True, name="multiParity"):
    """Parity rig with *multiple* weighted splines feeding each rider.

    Exercises the multi-spline path: trans/scale/twist weighting, quaternion
    slerp, and the X-axis twist quaternion. ``cv_hulls`` is a list of control
    hulls (one per spline); ``weights`` the matching blend weights.
    """
    if not cmds.pluginInfo("TwistSpline", q=True, loaded=True):
        raise RuntimeError("Load the C++ plugin first (cmds.loadPlugin('TwistSpline')).")
    if not cmds.pluginInfo("pyTwistSplinePlugin", q=True, loaded=True):
        raise RuntimeError("Load the Python plugin first.")

    py_specs, cpp_specs = [], []
    for hi, hull in enumerate(cv_hulls):
        cvs = [list(p) for p in hull]
        cv_locs, out_locs, in_locs = _make_controls(cvs, "{}_h{}_ctl".format(name, hi))
        py_s = _wire_spline("pyTwistSpline", "{}_h{}_pyShape".format(name, hi),
                            cv_locs, out_locs, in_locs, spread)
        cpp_s = _wire_spline("twistSpline", "{}_h{}_cppShape".format(name, hi),
                             cv_locs, out_locs, in_locs, spread)
        py_specs.append((py_s, weights[hi]))
        cpp_specs.append((cpp_s, weights[hi]))

    parent = None
    if use_parent:
        parent = cmds.createNode("transform", name=name + "_rigParent")
        cmds.setAttr(parent + ".translate", 1.5, -2.0, 0.75)
        cmds.setAttr(parent + ".rotate", 20.0, -35.0, 12.0)

    py_rider = _make_rider("pyRiderConstraint", name + "_pyRider",
                           py_specs, params, rider_globals, parent)
    cpp_rider = _make_rider("riderConstraint", name + "_cppRider",
                            cpp_specs, params, rider_globals, parent)

    return {"py_rider": py_rider, "cpp_rider": cpp_rider,
            "parent": parent, "nparams": len(params)}


def compare_riders(rig, tol=1e-6):
    """Compare every output transform of the py rider vs the C++ rider.

    Returns (max_dTranslate, max_dRotate_deg, max_dScale).
    """
    py, cpp, nparams = rig["py_rider"], rig["cpp_rider"], rig["nparams"]
    max_t = max_r = max_s = 0.0
    for i in range(nparams):
        for chan, store in (("translate", "t"), ("rotate", "r"), ("scale", "s")):
            a = cmds.getAttr("{}.outputs[{}].{}".format(py, i, chan))[0]
            b = cmds.getAttr("{}.outputs[{}].{}".format(cpp, i, chan))[0]
            d = max(abs(a[k] - b[k]) for k in range(3))
            if store == "t":
                max_t = max(max_t, d)
            elif store == "r":
                max_r = max(max_r, d)
            else:
                max_s = max(max_s, d)

    ok = max_t < tol and max_r < tol and max_s < tol
    print("py vs C++ rider | dT={:.3e} | dR(deg)={:.3e} | dS={:.3e} | {}".format(
        max_t, max_r, max_s, "OK" if ok else "MISMATCH"))
    return max_t, max_r, max_s


# ---------------------------------------------------------------------------
# Phase 3: parity for the twistTangent node
# ---------------------------------------------------------------------------

def _tan_loc(pos, rot, name):
    loc = cmds.spaceLocator(name=name)[0]
    cmds.xform(loc, worldSpace=True, translation=list(pos))
    if rot != (0.0, 0.0, 0.0):
        cmds.xform(loc, worldSpace=True, rotation=list(rot))
    return loc


def create_tangent_parity(prev_pos=(-3, 0, 0), cur_pos=(0, 0, 0), next_pos=(3, 1, 1),
                          in_pos=(1.2, 0.4, 0.0), cur_rot=(10, 25, -8),
                          auto=0.5, smooth=1.0, weight=1.0, backpoint=False,
                          endpoint=False, in_linear_target=(0.7, 0.5, 0.2),
                          use_parent=True, name="tanParity"):
    """Drive a pyTwistTangent and a C++ twistTangent with identical inputs."""
    for plug in ("TwistSpline", "pyTwistSplinePlugin"):
        if not cmds.pluginInfo(plug, q=True, loaded=True):
            raise RuntimeError("Load both plugins first (missing {}).".format(plug))

    prevT = _tan_loc(prev_pos, (0, 0, 0), name + "_prev")
    curT = _tan_loc(cur_pos, cur_rot, name + "_cur")
    nextT = _tan_loc(next_pos, (0, 0, 0), name + "_next")
    inT = _tan_loc(in_pos, (0, 0, 0), name + "_inTan")

    parent = None
    if use_parent:
        parent = cmds.createNode("transform", name=name + "_parent")
        cmds.setAttr(parent + ".translate", 0.8, -1.1, 0.3)
        cmds.setAttr(parent + ".rotate", -15.0, 22.0, 9.0)

    nodes = {}
    for key, node_type in (("py", "pyTwistTangent"), ("cpp", "twistTangent")):
        n = cmds.createNode(node_type, name="{}_{}".format(name, key))
        cmds.connectAttr(prevT + ".worldMatrix[0]", n + ".previousVertex")
        cmds.connectAttr(curT + ".worldMatrix[0]", n + ".currentVertex")
        cmds.connectAttr(nextT + ".worldMatrix[0]", n + ".nextVertex")
        cmds.connectAttr(inT + ".worldMatrix[0]", n + ".inTangent")
        if parent is not None:
            cmds.connectAttr(parent + ".worldInverseMatrix[0]", n + ".parentInverseMatrix")
        cmds.setAttr(n + ".auto", auto)
        cmds.setAttr(n + ".smooth", smooth)
        cmds.setAttr(n + ".weight", weight)
        cmds.setAttr(n + ".backpoint", backpoint)
        cmds.setAttr(n + ".endpoint", endpoint)
        cmds.setAttr(n + ".inLinearTarget", *in_linear_target, type="double3")
        nodes[key] = n

    nodes["transforms"] = [prevT, curT, nextT, inT, parent]
    return nodes


def compare_tangents(nodes, tol=1e-6):
    """Compare every output of the py twistTangent vs the C++ one."""
    py, cpp = nodes["py"], nodes["cpp"]
    worst = {}
    for chan in ("out", "smoothTan", "outLinearTarget", "outTwistUp"):
        a = cmds.getAttr("{}.{}".format(py, chan))[0]
        b = cmds.getAttr("{}.{}".format(cpp, chan))[0]
        worst[chan] = max(abs(a[k] - b[k]) for k in range(3))
    am = cmds.getAttr(py + ".outTwistMat")
    bm = cmds.getAttr(cpp + ".outTwistMat")
    worst["outTwistMat"] = max(abs(am[k] - bm[k]) for k in range(16))

    mx = max(worst.values())
    ok = mx < tol
    print("py vs C++ twistTangent | " +
          " | ".join("{}={:.2e}".format(k, v) for k, v in worst.items()) +
          " | {}".format("OK" if ok else "MISMATCH"))
    return worst


# ---------------------------------------------------------------------------
# Phase 3: parity for the twistMultiTangent node
# ---------------------------------------------------------------------------

def create_multi_tangent_parity(cv_positions, in_auto=0.7, out_auto=0.7,
                                in_smooth=0.8, out_smooth=0.8, in_weight=1.1,
                                out_weight=0.9, start_tension=2.0, end_tension=2.0,
                                closed=False, use_parent=True, name="mtanParity"):
    """Drive a pyTwistMultiTangent and a C++ twistMultiTangent identically.

    Vertex transforms get varied rotations (so vertMat row1/row3 and the
    local-space user tangents are non-trivial); user tangent controls sit at
    Catmull-Rom points. Non-1 auto/smooth/weights exercise every blend path.
    """
    for plug in ("TwistSpline", "pyTwistSplinePlugin"):
        if not cmds.pluginInfo(plug, q=True, loaded=True):
            raise RuntimeError("Load both plugins first (missing {}).".format(plug))

    cvs = [list(p) for p in cv_positions]
    n = len(cvs)
    out_tans, in_tans = _catmull_tangents(cvs)

    vert_locs, in_user_locs, out_user_locs = [], [], []
    for i in range(n):
        vert_locs.append(_tan_loc(cvs[i], (i * 7.0, -i * 5.0, i * 3.0),
                                  "{}_v{}".format(name, i)))
        in_user_locs.append(_tan_loc(in_tans[i], (0, 0, 0), "{}_iu{}".format(name, i)))
        out_user_locs.append(_tan_loc(out_tans[i], (0, 0, 0), "{}_ou{}".format(name, i)))

    parent = None
    if use_parent:
        parent = cmds.createNode("transform", name=name + "_parent")
        cmds.setAttr(parent + ".translate", -0.6, 1.3, -0.9)
        cmds.setAttr(parent + ".rotate", 12.0, -28.0, 17.0)

    nodes = {}
    for key, ntype in (("py", "pyTwistMultiTangent"), ("cpp", "twistMultiTangent")):
        node = cmds.createNode(ntype, name="{}_{}".format(name, key))
        cmds.setAttr(node + ".startTension", start_tension)
        cmds.setAttr(node + ".endTension", end_tension)
        cmds.setAttr(node + ".closed", closed)
        for i in range(n):
            vd = "{}.vertData[{}]".format(node, i)
            cmds.connectAttr(vert_locs[i] + ".worldMatrix[0]", vd + ".vertMat")
            cmds.connectAttr(in_user_locs[i] + ".worldMatrix[0]", vd + ".inTanMat")
            cmds.connectAttr(out_user_locs[i] + ".worldMatrix[0]", vd + ".outTanMat")
            if parent is not None:
                for pim in ("inParentInverseMatrix", "outParentInverseMatrix",
                            "twistParentInverseMatrix"):
                    cmds.connectAttr(parent + ".worldInverseMatrix[0]", vd + "." + pim)
            cmds.setAttr(vd + ".inTanWeight", in_weight)
            cmds.setAttr(vd + ".outTanWeight", out_weight)
            cmds.setAttr(vd + ".inSmooth", in_smooth)
            cmds.setAttr(vd + ".outSmooth", out_smooth)
            cmds.setAttr(vd + ".inAuto", in_auto)
            cmds.setAttr(vd + ".outAuto", out_auto)
        nodes[key] = node

    nodes["nverts"] = n
    return nodes


def compare_multi_tangents(nodes, tol=1e-6):
    """Compare every per-vertex output of the py vs C++ multi-tangent node."""
    py, cpp, n = nodes["py"], nodes["cpp"], nodes["nverts"]
    worst = {}

    def _bump(key, val):
        worst[key] = max(worst.get(key, 0.0), val)

    for i in range(n):
        for chan in ("inVertTan", "outVertTan", "twistUp"):
            a = cmds.getAttr("{}.vertTans[{}].{}".format(py, i, chan))[0]
            b = cmds.getAttr("{}.vertTans[{}].{}".format(cpp, i, chan))[0]
            _bump(chan, max(abs(a[k] - b[k]) for k in range(3)))
        for chan in ("inTanLen", "outTanLen"):
            a = cmds.getAttr("{}.vertTans[{}].{}".format(py, i, chan))
            b = cmds.getAttr("{}.vertTans[{}].{}".format(cpp, i, chan))
            _bump(chan, abs(a - b))
        am = cmds.getAttr("{}.vertTans[{}].twistMat".format(py, i))
        bm = cmds.getAttr("{}.vertTans[{}].twistMat".format(cpp, i))
        _bump("twistMat", max(abs(am[k] - bm[k]) for k in range(16)))

    mx = max(worst.values())
    ok = mx < tol
    print("py vs C++ multiTangent | " +
          " | ".join("{}={:.2e}".format(k, v) for k, v in sorted(worst.items())) +
          " | {}".format("OK" if ok else "MISMATCH"))
    return worst
