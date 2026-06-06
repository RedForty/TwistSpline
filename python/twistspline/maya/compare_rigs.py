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
