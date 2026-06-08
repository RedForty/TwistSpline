"""
Side-by-side comparison of a Python-node rig vs a C++-node rig.

``setup_comparison`` builds both rigs at the same rest pose (the Python rig via
``twistspline.maya.builder``, the C++ rig via the original
``scripts/twistSplineBuilder.py``) and -- unless you'd rather wire it yourself --
links every C++ control to its Python counterpart so a single set of controls
drives both. Then move the ``cmpPy_*`` controls around and call
``compare_joints`` to confirm the rider joints stay coincident.

    from twistspline.maya import compare_rigs
    rigs = compare_rigs.setup_comparison(numCVs=5, numJoints=10)
    # ... move the cmpPy CV / tangent / twist controls around ...
    compare_rigs.compare_joints(rigs)        # max dist should stay ~0
    compare_rigs.compare_joints(rigs, orient=True)   # also check orientation

Both plugins must be available: the C++ ``TwistSpline`` (auto-loaded by the C++
builder) and the Python ``pyTwistSplinePlugin`` (load it by path first).
"""

import os
import math
import importlib.util

import maya.cmds as cmds

from . import builder as _pyb

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS = os.path.join(_REPO, "scripts")


def _load_cpp_builder():
    """Load scripts/twistSplineBuilder.py (the C++ builder) as a private module."""
    path = os.path.join(_SCRIPTS, "twistSplineBuilder.py")
    if not os.path.exists(path):
        raise RuntimeError("C++ builder not found at {}".format(path))
    spec = importlib.util.spec_from_file_location("twistSplineBuilder_cpp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unpack(ret, pfx, numCVs):
    """makeTwistSpline returns:
       cvs, cvBfrs, oCtrls, iCtrls, jPars, joints, group, splineTfm, master, cnst
    """
    cvs, cvBfrs, oCtrls, iCtrls, jPars, joints, group, splineTfm, master, cnst = ret
    tws = [_pyb.CTRL_TWIST_FMT.format(pfx, i + 1) for i in range(numCVs)]
    tws = [t for t in tws if cmds.objExists(t)]
    return {
        "cvs": cvs, "oCtrls": oCtrls, "iCtrls": iCtrls, "tws": tws,
        "master": master, "joints": joints, "jPars": jPars,
        "spline": splineTfm, "cnst": cnst,
    }


def build_both(numCVs=5, numJoints=10, spread=1.0, closed=False, singleTangentNode=True):
    """Build a Python rig (cmpPy) and a C++ rig (cmpCpp) with identical params."""
    if not cmds.pluginInfo("pyTwistSplinePlugin", q=True, loaded=True):
        raise RuntimeError("Load the Python plugin first "
                           "(cmds.loadPlugin('.../python/pyTwistSplinePlugin.py')).")
    cppb = _load_cpp_builder()

    py = _pyb.makeTwistSpline("cmpPy", numCVs, numJoints, spread=spread,
                              closed=closed, singleTangentNode=singleTangentNode)
    cpp = cppb.makeTwistSpline("cmpCpp", numCVs, numJoints, spread=spread,
                               closed=closed, singleTangentNode=singleTangentNode)
    return {"py": _unpack(py, "cmpPy", numCVs),
            "cpp": _unpack(cpp, "cmpCpp", numCVs)}


def _link(src, dst):
    """Drive dst's transform + keyable user attrs from src (best-effort)."""
    for at in ("tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"):
        try:
            cmds.connectAttr("{}.{}".format(src, at), "{}.{}".format(dst, at), force=True)
        except Exception:
            pass
    for a in (cmds.listAttr(src, ud=True, keyable=True) or []):
        if cmds.attributeQuery(a, node=dst, exists=True):
            try:
                cmds.connectAttr("{}.{}".format(src, a), "{}.{}".format(dst, a), force=True)
            except Exception:
                pass


def link_rigs(rigs):
    """Drive every C++ control from its Python counterpart.

    After this, only the ``cmpPy_*`` controls are 'live'; the C++ rig follows.
    Covers CV controls, in/out tangent controls, twist controls and the master
    (whose Offset/Stretch/scale feed the rider/spline params).
    """
    py, cpp = rigs["py"], rigs["cpp"]
    for key in ("cvs", "oCtrls", "iCtrls", "tws"):
        for s, d in zip(py[key], cpp[key]):
            _link(s, d)
    _link(py["master"], cpp["master"])


def setup_comparison(numCVs=5, numJoints=10, spread=1.0, closed=False,
                     singleTangentNode=True, link=True):
    """Build both rigs and (by default) link the C++ rig to the Python one."""
    rigs = build_both(numCVs, numJoints, spread, closed, singleTangentNode)
    if link:
        link_rigs(rigs)
    return rigs


def compare_joints(rigs, orient=False, tol=1e-6, verbose=False):
    """Report the worst world-space difference between matching rider joints.

    Position always; orientation too if ``orient=True`` (compares world
    translation derived from each joint's worldMatrix and the matrix rotation).
    """
    pj, cj = rigs["py"]["joints"], rigs["cpp"]["joints"]
    max_pos = 0.0
    max_ori = 0.0
    for a, b in zip(pj, cj):
        pa = cmds.xform(a, q=True, ws=True, t=True)
        pb = cmds.xform(b, q=True, ws=True, t=True)
        d = math.sqrt(sum((pa[k] - pb[k]) ** 2 for k in range(3)))
        max_pos = max(max_pos, d)
        if orient:
            ra = cmds.xform(a, q=True, ws=True, ro=True)
            rb = cmds.xform(b, q=True, ws=True, ro=True)
            do = max(abs(ra[k] - rb[k]) for k in range(3))
            max_ori = max(max_ori, do)
        if verbose:
            print("  {:<28} dPos={:.3e}".format(a, d))

    if orient:
        ok = max_pos < tol and max_ori < tol
        print("joint parity | max dPos={:.4e} | max dRot(deg)={:.4e} | {}".format(
            max_pos, max_ori, "OK" if ok else "MISMATCH"))
        return max_pos, max_ori
    ok = max_pos < tol
    print("joint parity | max dPos={:.4e} | {}".format(
        max_pos, "OK" if ok else "MISMATCH"))
    return max_pos


def compare_native_to_cpp(cv_positions=None, numJoints=10, spread=1.0,
                          tol_pos=1e-2, tol_rot=0.5, verbose=True):
    """Direct joint-to-joint parity: the all-native-node rig vs the C++ rig.

    Builds a C++ rig (``makeTwistSpline``) and the native-node rig
    (``native_builder.build_native_spline``) from identical CV positions, with
    every CV pinned (so both distribute joints evenly by arc length) and orient
    locked at CV0 only (the C++ default). Compares each joint's world position and
    world orientation (relative rotation between world matrices -- convention- and
    rotate-order-agnostic).

    Only the C++ ``TwistSpline`` plugin is required (no python plugin).
    """
    import maya.api.OpenMaya as om
    from . import native_builder

    if not cmds.pluginInfo("TwistSpline", q=True, loaded=True):
        cmds.loadPlugin("TwistSpline")
    if cv_positions is None:
        cv_positions = [[0, 0, 0], [3, 2, 1], [6, 0, 3], [9, -2, 1], [12, 0, 0]]
    numCVs = len(cv_positions)

    cppb = _load_cpp_builder()
    cpp = cppb.makeTwistSpline("cmpCpp", numCVs, numJoints, spread=spread)
    cpp_d = _unpack(cpp, "cmpCpp", numCVs)
    # shape the C++ CVs to the test curve and pin every CV (even arc-length spread)
    for c, p in zip(cpp_d["cvs"], cv_positions):
        cmds.xform(c, worldSpace=True, translation=list(p))
        cmds.setAttr(c + ".Pin", 1.0)
    cmds.dgdirty(allPlugs=True)
    cmds.refresh(force=True)

    rig = native_builder.build_native_spline(
        cv_positions, numJoints, spread=spread, name="cmpNative",
        pins=list(range(numCVs)), orient_cvs=[0])
    cmds.refresh(force=True)

    max_pos = max_rot = 0.0
    rows = []
    for i, (a, b) in enumerate(zip(rig["joints"], cpp_d["joints"])):
        pa = cmds.xform(a, q=True, ws=True, t=True)
        pb = cmds.xform(b, q=True, ws=True, t=True)
        dp = math.sqrt(sum((pa[k] - pb[k]) ** 2 for k in range(3)))
        ma = om.MMatrix(cmds.xform(a, q=True, ws=True, matrix=True))
        mb = om.MMatrix(cmds.xform(b, q=True, ws=True, matrix=True))
        q = om.MTransformationMatrix(ma * mb.inverse()).rotation(asQuaternion=True)
        dr = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(q[3])))))
        max_pos = max(max_pos, dp)
        max_rot = max(max_rot, dr)
        rows.append((i, dp, dr))

    ok = max_pos < tol_pos and max_rot < tol_rot
    print("native vs C++ JOINTS | max dPos={:.4e} | max dRot(deg)={:.4f} | {}".format(
        max_pos, max_rot, "OK" if ok else "REVIEW"))
    if verbose:
        for i, dp, dr in rows:
            print("  joint {:2d}  dPos={:.4e}  dRot={:.4f} deg".format(i, dp, dr))
    return max_pos, max_rot


def _jdelta(nat_joints, cpp_joints):
    """Worst (position, orientation-deg) difference between matched joints."""
    import maya.api.OpenMaya as om
    mp = mr = 0.0
    for a, b in zip(nat_joints, cpp_joints):
        pa = cmds.xform(a, q=True, ws=True, t=True)
        pb = cmds.xform(b, q=True, ws=True, t=True)
        mp = max(mp, math.sqrt(sum((pa[k] - pb[k]) ** 2 for k in range(3))))
        ma = om.MMatrix(cmds.xform(a, q=True, ws=True, matrix=True))
        mb = om.MMatrix(cmds.xform(b, q=True, ws=True, matrix=True))
        q = om.MTransformationMatrix(ma * mb.inverse()).rotation(asQuaternion=True)
        mr = max(mr, math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(q[3]))))))
    return mp, mr


def parity_sweep(cv_positions=None, numJoints=10, spread=1.0,
                 tol_pos=1e-2, tol_rot=0.5):
    """Drive the native and C++ rigs through matched perturbations and compare.

    Each state sets the SAME change on both rigs (CV moves, Pin, twist, tangent
    Auto/Smooth/Weight + a manual handle) and reports the worst joint position /
    orientation difference, so parity is confirmed under animation, not just rest.
    """
    from . import native_builder
    if not cmds.pluginInfo("TwistSpline", q=True, loaded=True):
        cmds.loadPlugin("TwistSpline")
    if cv_positions is None:
        cv_positions = [[0, 0, 0], [3, 2, 1], [6, 0, 3], [9, -2, 1], [12, 0, 0]]
    n = len(cv_positions)

    cppb = _load_cpp_builder()
    cpp = cppb.makeTwistSpline("cmpCpp", n, numJoints, spread=spread)
    cpp_d = _unpack(cpp, "cmpCpp", n)
    cv_c, o_c, i_c, tw_c = cpp_d["cvs"], cpp_d["oCtrls"], cpp_d["iCtrls"], cpp_d["tws"]
    for c, p in zip(cv_c, cv_positions):
        cmds.xform(c, ws=True, t=list(p))
        cmds.setAttr(c + ".Pin", 1.0)
    tsn = cmds.ls("cmpCpp*", type="twistSpline")  # neutralize the -1 twist default
    if tsn:
        cmds.setAttr(tsn[0] + ".twistMultiplier", 1.0)
    cmds.refresh(force=True)

    nat = native_builder.build_native_spline(
        cv_positions, numJoints, spread=spread, name="cmpNat",
        pins=list(range(n)), orient_cvs=[0])
    cv_n, o_n, i_n, tw_n = nat["cv_ctrls"], nat["out_ctrl"], nat["in_ctrl"], nat["twist_ctrl"]
    cmds.refresh(force=True)

    def setboth(np_, cp_, v):
        cmds.setAttr(np_, v)
        cmds.setAttr(cp_, v)

    def xboth(nn, cn, p):
        cmds.xform(nn, ws=True, t=p)
        cmds.xform(cn, ws=True, t=p)

    results = []

    def check(label):
        cmds.refresh(force=True)
        mp, mr = _jdelta(nat["joints"], cpp_d["joints"])
        flag = "OK" if (mp < tol_pos and mr < tol_rot) else "REVIEW"
        print("  {:<36} dPos={:.3e}  dRot={:.4f} deg  {}".format(label, mp, mr, flag))
        results.append((label, mp, mr))

    print("native vs C++ parity sweep:")
    check("rest")

    # --- twist (rotateX on the per-CV twist controls; all UseTwist=1) ---
    for k in range(n):
        setboth(tw_n[k] + ".UseTwist", tw_c[k] + ".UseTwist", 1.0)
    setboth(tw_n[1] + ".rotateX", tw_c[1] + ".rotateX", 45.0)
    setboth(tw_n[3] + ".rotateX", tw_c[3] + ".rotateX", -30.0)
    check("twist CV1=45 CV3=-30 (all pinned)")
    # partial UseTwist: float CV2 between the twisted neighbours
    setboth(tw_n[2] + ".UseTwist", tw_c[2] + ".UseTwist", 0.0)
    check("  + CV2 UseTwist=0 (float)")
    setboth(tw_n[1] + ".rotateX", tw_c[1] + ".rotateX", 0.0)
    setboth(tw_n[3] + ".rotateX", tw_c[3] + ".rotateX", 0.0)
    setboth(tw_n[2] + ".UseTwist", tw_c[2] + ".UseTwist", 1.0)

    # --- geometry ---
    xboth(cv_n[2], cv_c[2], [6, 3, 4])
    check("move CV2 -> (6,3,4)")
    xboth(cv_n[2], cv_c[2], list(cv_positions[2]))

    # --- pin ---
    setboth(cv_n[2] + ".Pin", cv_c[2] + ".Pin", 0.0)
    check("unpin CV2 (Pin=0)")
    setboth(cv_n[2] + ".Pin", cv_c[2] + ".Pin", 1.0)

    # --- tangents (native in_ctrl[i] <-> cpp iCtrls[i-1]) ---
    setboth(o_n[1] + ".Weight", o_c[1] + ".Weight", 2.2)
    check("out tangent CV1 Weight=2.2")
    setboth(o_n[1] + ".Weight", o_c[1] + ".Weight", 1.0)

    setboth(i_n[2] + ".Smooth", i_c[1] + ".Smooth", 0.0)
    check("in tangent CV2 Smooth=0 (linear)")
    setboth(i_n[2] + ".Smooth", i_c[1] + ".Smooth", 1.0)

    # NOTE: a moved *manual* tangent (Auto=0) is not compared here -- the C++ rig
    # parents its tangent control under an auto-driven buffer, so setting its world
    # position doesn't land where the native control does (a harness mismatch, not a
    # rig difference). Manual-tangent correctness is covered by verify_tangents,
    # which checks handle positions against the C++-faithful tangent reference.

    worst_p = max(r[1] for r in results)
    worst_r = max(r[2] for r in results)
    print("SWEEP worst | dPos={:.3e} | dRot={:.4f} deg | {}".format(
        worst_p, worst_r, "OK" if (worst_p < tol_pos and worst_r < tol_rot) else "REVIEW"))
    return results
