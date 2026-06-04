"""
Rung 4 verification: prove the Python kernel matches the compiled C++ plugin.

This builds a :class:`twistspline.core.TwistSpline` from the *inputs* of an
existing C++ ``twistSpline`` node, samples it at the same parameters a
``riderConstraint`` uses, and compares the resulting poses against the rider's
actual outputs. It can also dump everything to JSON so the comparison becomes a
repeatable regression test that doesn't need the scene rebuilt.

This file DOES use ``maya.cmds`` / ``maya.api.OpenMaya`` -- it is the bridge
between Maya and the dependency-free kernel, not part of the kernel itself.

Usage (in the Maya Script Editor, with a TwistSpline rig built via
``twistSplineBuilder``)::

    import sys; sys.path.append("/path/to/TwistSpline/python")
    import maya_parity
    maya_parity.compare("twistSpline1", "riderConstraint1")
    maya_parity.dump_golden("twistSpline1", "riderConstraint1", "/tmp/golden.json")

NOTE on the rider transform: this replicates the single-spline path of
``riderConstraint::compute``. For the cleanest first comparison, leave the rider
at defaults (globalSpread=1, globalOffset=0, useCycle off) or just trust the
replicated transform below.
"""

import json
from math import radians, fmod

import maya.cmds as cmds
import maya.api.OpenMaya as om

from twistspline.core import TwistSpline
from twistspline import _vmath as vm

LUT_STEPS = 20  # must match the C++ default (TwistSpline() sets lutSteps = 20)


# ---------------------------------------------------------------------------
# Node-name resolution
# ---------------------------------------------------------------------------

def find_nodes():
    """Print the twistSpline shapes and riderConstraints in the scene."""
    splines = cmds.ls(type="twistSpline") or []
    riders = cmds.ls(type="riderConstraint") or []
    print("twistSpline shapes : {}".format(splines))
    print("riderConstraints   : {}".format(riders))
    return splines, riders


def _resolve_spline(name):
    """Resolve a transform/shape name to the twistSpline *shape* node.

    The twistSpline node is a locator, so a name like 'twistSpline1' is usually
    the transform -- the attributes live on its shape child.
    """
    if not cmds.objExists(name):
        raise ValueError("No node named '{}'. twistSpline shapes in scene: {}"
                         .format(name, cmds.ls(type="twistSpline")))
    if cmds.nodeType(name) == "twistSpline":
        return name
    shapes = cmds.listRelatives(name, shapes=True, type="twistSpline",
                                fullPath=False) or []
    if shapes:
        return shapes[0]
    raise ValueError("'{}' is not a twistSpline and has no twistSpline shape "
                     "child. twistSpline shapes in scene: {}"
                     .format(name, cmds.ls(type="twistSpline")))


def _resolve_rider(name):
    """Resolve to a riderConstraint node (it's a DG node, not a DAG shape)."""
    if not cmds.objExists(name):
        raise ValueError("No node named '{}'. riderConstraints in scene: {}"
                         .format(name, cmds.ls(type="riderConstraint")))
    if cmds.nodeType(name) == "riderConstraint":
        return name
    # Tolerate a transform that happens to parent one (rare).
    rels = cmds.listRelatives(name, shapes=True, type="riderConstraint") or []
    if rels:
        return rels[0]
    raise ValueError("'{}' is not a riderConstraint. riderConstraints in scene: "
                     "{}".format(name, cmds.ls(type="riderConstraint")))


# ---------------------------------------------------------------------------
# Reading a C++ twistSpline node back into a Python TwistSpline
# ---------------------------------------------------------------------------

def _decompose(flat16):
    """Flat 16-float Maya matrix -> (translation[3], quat[w,x,y,z], scale[3])."""
    m = om.MMatrix(flat16)
    tm = om.MTransformationMatrix(m)
    t = tm.translation(om.MSpace.kWorld)
    q = tm.rotation(asQuaternion=True)
    s = tm.scale(om.MSpace.kObject)
    return [t.x, t.y, t.z], [q.w, q.x, q.y, q.z], [s[0], s[1], s[2]]


def build_spline_from_node(spline_node, lut_steps=LUT_STEPS):
    """Reconstruct a Python TwistSpline from a C++ ``twistSpline`` node's inputs.

    Mirrors the assembly in twistSplineNode.cpp:308-379 exactly (interleaving
    inTangent / controlVertex / outTangent, dropping the leading in-tangent and
    trailing out-tangent, and applying the well-posing guards).
    """
    spline_node = _resolve_spline(spline_node)
    scale_comp = cmds.getAttr(spline_node + ".scaleCompensation")
    twist_mul = cmds.getAttr(spline_node + ".twistMultiplier")
    ecount = cmds.getAttr(spline_node + ".vertexData", size=True)

    points, scales, quats = [], [], []
    lock_positions, lock_vals, twist_lock, user_twist, orient_lock = [], [], [], [], []
    got_locks = got_oris = False

    for i in range(ecount):
        vd = "{}.vertexData[{}]".format(spline_node, i)

        lock_positions.append(cmds.getAttr(vd + ".paramValue") * scale_comp)
        lk = cmds.getAttr(vd + ".paramWeight")
        lock_vals.append(lk)
        got_locks = got_locks or lk > 0.0

        twist_lock.append(cmds.getAttr(vd + ".twistWeight"))
        user_twist.append(cmds.getAttr(vd + ".twistValue") * twist_mul)

        ori = cmds.getAttr(vd + ".useOrient")
        orient_lock.append(ori)
        got_oris = got_oris or ori > 0.0

        if i > 0:
            t, q, s = _decompose(cmds.getAttr(vd + ".inTangent"))
            points.append(t); scales.append(s); quats.append(q)

        t, q, s = _decompose(cmds.getAttr(vd + ".controlVertex"))
        points.append(t); scales.append(s); quats.append(q)

        t, q, s = _decompose(cmds.getAttr(vd + ".outTangent"))
        points.append(t); scales.append(s); quats.append(q)

    # Drop the trailing out-tangent group (it's past the end).
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

    spline = TwistSpline(lut_steps=lut_steps)
    spline.set_verts(points, scales, quats, lock_positions, lock_vals,
                     user_twist, twist_lock, orient_lock)
    return spline


# ---------------------------------------------------------------------------
# Replicating the rider's parameter transform (single-spline path)
# ---------------------------------------------------------------------------

def _rider_globals(rider):
    g = {
        "scaleComp": cmds.getAttr(rider + ".scaleCompensation"),
        "gOffset": cmds.getAttr(rider + ".globalOffset"),
        "gSpread": cmds.getAttr(rider + ".globalSpread"),
        "useCycle": cmds.getAttr(rider + ".useCycle"),
        "doNorm": cmds.getAttr(rider + ".normalize"),
        "normVal": cmds.getAttr(rider + ".normValue"),
        "order": cmds.getAttr(rider + ".rotateOrder"),
    }
    g["gOffset"] *= g["scaleComp"]
    g["gSpread"] *= g["scaleComp"]
    g["normVal"] *= g["scaleComp"]
    return g


def _transform_param(p, g, mp, mrmp):
    """Port of the per-param remap in riderConstraint.cpp:563-580."""
    p = p * g["gSpread"] + g["gOffset"]
    if g["doNorm"] > 0.0:
        p = g["doNorm"] * (p * mp / g["normVal"]) + (1.0 - g["doNorm"]) * p
    if g["useCycle"]:
        if p >= 0.0:
            p = fmod(p, mrmp)
        else:
            p = mrmp - fmod(-p, mrmp)
    return p


def _python_pose(spline, p, inv_par_flat, order):
    """Sample the kernel at ``p`` and build the same (translate, rotate-quat)
    the rider would output, in the param's parent space."""
    f = spline.matrix_at_param(p, twisted=True)
    # Rider matrix rows are (tan, norm, binorm) -> (X, Y, Z), translation last.
    world = om.MMatrix([
        f.tan[0], f.tan[1], f.tan[2], 0.0,
        f.norm[0], f.norm[1], f.norm[2], 0.0,
        f.binorm[0], f.binorm[1], f.binorm[2], 0.0,
        f.tran[0], f.tran[1], f.tran[2], 1.0,
    ])
    local = world * om.MMatrix(inv_par_flat)
    tm = om.MTransformationMatrix(local)
    t = tm.translation(om.MSpace.kWorld)
    q = tm.rotation(asQuaternion=True)
    return [t.x, t.y, t.z], [q.w, q.x, q.y, q.z]


def _rider_pose(rider, idx, order):
    """The rider's actual output pose for param ``idx`` as (translate, quat)."""
    out = "{}.outputs[{}]".format(rider, idx)
    t = cmds.getAttr(out + ".translate")[0]
    r = cmds.getAttr(out + ".rotate")[0]  # degrees
    eul = om.MEulerRotation(radians(r[0]), radians(r[1]), radians(r[2]), order)
    q = eul.asQuaternion()
    return [t[0], t[1], t[2]], [q.w, q.x, q.y, q.z]


def _iter_params(rider):
    """Yield (idx, param_value, parentInverseMatrix_flat) for each rider param."""
    count = cmds.getAttr(rider + ".params", size=True)
    for idx in range(count):
        ps = "{}.params[{}]".format(rider, idx)
        try:
            p = cmds.getAttr(ps + ".param")
        except Exception:
            continue
        pim = cmds.getAttr(ps + ".parentInverseMatrix")
        yield idx, p, pim


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def compare(spline_node, rider_node, tol_pos=1e-5, tol_ang_deg=1e-3, verbose=True):
    """Compare the Python kernel against the C++ rider. Returns (max_pos_err,
    max_ang_err_deg) and prints a per-param report when ``verbose``."""
    rider_node = _resolve_rider(rider_node)
    spline = build_spline_from_node(spline_node)
    g = _rider_globals(rider_node)

    mp = cmds.getAttr(rider_node + ".inputSplines[0].endParam")
    mrmp = cmds.getAttr(rider_node + ".inputSplines[0].splineLength")
    if mrmp == 0.0:
        mp = spline.lock_positions[-1]
        mrmp = spline.remap[-1]

    max_pos = 0.0
    max_ang = 0.0
    n = 0
    for idx, raw_p, pim in _iter_params(rider_node):
        p = _transform_param(raw_p, g, mp, mrmp)
        py_t, py_q = _python_pose(spline, p, pim, g["order"])
        cpp_t, cpp_q = _rider_pose(rider_node, idx, g["order"])

        pos_err = vm.length(vm.sub(py_t, cpp_t))
        ang_err = vm.quat_angle_between(py_q, cpp_q)
        ang_deg = ang_err * 180.0 / 3.141592653589793
        max_pos = max(max_pos, pos_err)
        max_ang = max(max_ang, ang_deg)
        n += 1
        if verbose:
            flag = "" if (pos_err < tol_pos and ang_deg < tol_ang_deg) else "  <-- MISMATCH"
            print("  param[{:>3}] p={:8.4f}  dPos={:.3e}  dAng={:.3e} deg{}".format(
                idx, p, pos_err, ang_deg, flag))

    print("-" * 60)
    ok = max_pos < tol_pos and max_ang < tol_ang_deg
    print("{} riders | max dPos = {:.3e} | max dAng = {:.3e} deg | {}".format(
        n, max_pos, max_ang, "PARITY OK" if ok else "PARITY FAILED"))
    return max_pos, max_ang


def dump_golden(spline_node, rider_node, path):
    """Capture inputs + C++ outputs to a JSON golden file for offline regression."""
    spline_node = _resolve_spline(spline_node)
    rider_node = _resolve_rider(rider_node)
    scale_comp = cmds.getAttr(spline_node + ".scaleCompensation")
    twist_mul = cmds.getAttr(spline_node + ".twistMultiplier")
    ecount = cmds.getAttr(spline_node + ".vertexData", size=True)

    verts = []
    for i in range(ecount):
        vd = "{}.vertexData[{}]".format(spline_node, i)
        verts.append({
            "controlVertex": list(cmds.getAttr(vd + ".controlVertex")),
            "inTangent": list(cmds.getAttr(vd + ".inTangent")),
            "outTangent": list(cmds.getAttr(vd + ".outTangent")),
            "paramValue": cmds.getAttr(vd + ".paramValue"),
            "paramWeight": cmds.getAttr(vd + ".paramWeight"),
            "twistValue": cmds.getAttr(vd + ".twistValue"),
            "twistWeight": cmds.getAttr(vd + ".twistWeight"),
            "useOrient": cmds.getAttr(vd + ".useOrient"),
        })

    riders = []
    for idx, p, pim in _iter_params(rider_node):
        out = "{}.outputs[{}]".format(rider_node, idx)
        riders.append({
            "index": idx,
            "param": p,
            "parentInverseMatrix": list(pim),
            "translate": list(cmds.getAttr(out + ".translate")[0]),
            "rotate": list(cmds.getAttr(out + ".rotate")[0]),
        })

    data = {
        "spline": {
            "node": spline_node,
            "scaleCompensation": scale_comp,
            "twistMultiplier": twist_mul,
            "lutSteps": LUT_STEPS,
            "vertexData": verts,
        },
        "rider": {
            "node": rider_node,
            "globals": _rider_globals(rider_node),
            "endParam": cmds.getAttr(rider_node + ".inputSplines[0].endParam"),
            "splineLength": cmds.getAttr(rider_node + ".inputSplines[0].splineLength"),
            "params": riders,
        },
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print("Wrote golden file: {} ({} verts, {} riders)".format(
        path, len(verts), len(riders)))
    return path
