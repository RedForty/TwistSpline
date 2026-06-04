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


def create_py_spline(cv_positions, spread=3.0, name="pyTwistSpline"):
    """Create a pyTwistSpline node + CV/tangent locators, wired up.

    Returns (shape, cv_locators, tangent_locators).
    """
    if not cmds.pluginInfo("pyTwistSplinePlugin", q=True, loaded=True):
        raise RuntimeError("Load the plugin first: "
                           "cmds.loadPlugin('.../python/pyTwistSplinePlugin.py')")

    cvs = [list(p) for p in cv_positions]
    n = len(cvs)
    out_tans, in_tans = _catmull_tangents(cvs)

    shape = cmds.createNode("pyTwistSpline", name=name + "Shape")

    cv_locs, out_locs, in_locs = [], [], []

    def _loc(pos, nm):
        loc = cmds.spaceLocator(name=nm)[0]
        cmds.xform(loc, worldSpace=True, translation=pos)
        return loc

    for i in range(n):
        cv_locs.append(_loc(cvs[i], "{}_cv{}".format(name, i)))
    for i in range(n - 1):
        out_locs.append(_loc(out_tans[i], "{}_out{}".format(name, i)))
        in_locs.append(_loc(in_tans[i + 1], "{}_in{}".format(name, i + 1)))

    for i in range(n):
        vd = "{}.vertexData[{}]".format(shape, i)
        cmds.connectAttr(cv_locs[i] + ".worldMatrix[0]", vd + ".controlVertex")
        cmds.setAttr(vd + ".paramValue", i * spread)
        if i < n - 1:
            cmds.connectAttr(out_locs[i] + ".worldMatrix[0]", vd + ".outTangent")
        if i > 0:
            cmds.connectAttr(in_locs[i - 1] + ".worldMatrix[0]", vd + ".inTangent")

    # Well-pose like the C++ builder: pin the first param + twist.
    cmds.setAttr(shape + ".vertexData[0].paramWeight", 1.0)
    cmds.setAttr(shape + ".vertexData[0].twistWeight", 1.0)
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
