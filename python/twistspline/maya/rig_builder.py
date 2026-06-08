"""Unified TwistSpline rig builder -- one entry point, pick the solver backend.

    from twistspline.maya import rig_builder
    rig = rig_builder.build("L_tail", cv_positions=[[0,0,0],[3,2,1],...],
                            num_joints=20, rig_type="native")

``rig_type``:
    "native"  -- 100% stock Maya nodes, no plugin (twistspline.maya.native_builder)
    "cpp"     -- the compiled C++ TwistSpline plugin (scripts/twistSplineBuilder.py)
    "python"  -- the pyTwistSplinePlugin scripted nodes (twistspline.maya.builder)

All three are feature-parity. The cpp and python backends already share ONE
control rig (``builder.mkTwistSplineControllers`` -- CV / twist / tangent / master
controls); the native backend is being moved onto the same controls so that any
improvement to the control rig is inherited by every backend.

Every backend returns the same normalized dict:
    rig_type, group, cv_ctrls, twist_ctrl, out_ctrl, in_ctrl, joints, curve, master
"""

import os
import importlib.util

import maya.cmds as cmds

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_NATIVE = ("native", "node", "nodes", "stock")
_CPP = ("cpp", "c++", "plugin")
_PYTHON = ("python", "py")


def _cpp_builder():
    """Load scripts/twistSplineBuilder.py (the C++ builder) as a private module."""
    path = os.path.join(_REPO, "scripts", "twistSplineBuilder.py")
    if not os.path.exists(path):
        raise RuntimeError("C++ builder not found at {}".format(path))
    spec = importlib.util.spec_from_file_location("twistSplineBuilder_cpp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_cvs(num_cvs, spread):
    """The production builders' default CV layout (a straight line on X)."""
    return [[i * 3.0 * spread, 0.0, 0.0] for i in range(num_cvs)]


def _from_production(rig_type, name, ret, cv_positions):
    """Normalize a makeTwistSpline (cpp/python) return into the common dict."""
    from .builder import CTRL_TWIST_FMT
    cvs, cvBfrs, oCtrls, iCtrls, jPars, joints, group, splineTfm, master, cnst = ret
    for c, p in zip(cvs, cv_positions):          # shape the hull to the request
        cmds.xform(c, worldSpace=True, translation=list(p))
    tws = [CTRL_TWIST_FMT.format(name, i + 1) for i in range(len(cvs))]
    tws = [t for t in tws if cmds.objExists(t)]
    return {"rig_type": rig_type, "group": group, "cv_ctrls": cvs,
            "twist_ctrl": tws, "out_ctrl": oCtrls, "in_ctrl": iCtrls,
            "joints": joints, "curve": splineTfm, "master": master, "cnst": cnst}


def build(name, cv_positions=None, num_cvs=None, num_joints=10, rig_type="native",
          spread=1.0, closed=False):
    """Build a TwistSpline rig of the chosen backend. Returns a normalized dict.

    Provide either ``cv_positions`` (a list of [x,y,z]) or ``num_cvs`` (CVs are
    then laid out on X by ``spread``, the production default).
    """
    rt = rig_type.lower()
    if cv_positions is None:
        cv_positions = _default_cvs(num_cvs or 5, spread)
    num_cvs = len(cv_positions)

    if rt in _NATIVE:
        from . import native_builder
        rig = native_builder.build_native_spline(
            cv_positions, num_joints, spread=spread, name=name)
        rig["rig_type"] = "native"
        rig.setdefault("master", None)
        rig.setdefault("group", rig.get("grp"))
        return rig

    if rt in _PYTHON:
        from . import builder as _b
        ret = _b.makeTwistSpline(name, num_cvs, num_joints, spread=spread, closed=closed)
        return _from_production("python", name, ret, cv_positions)

    if rt in _CPP:
        ret = _cpp_builder().makeTwistSpline(
            name, num_cvs, num_joints, spread=spread, closed=closed)
        return _from_production("cpp", name, ret, cv_positions)

    raise ValueError("rig_type must be one of native / cpp / python, got {!r}".format(rig_type))
