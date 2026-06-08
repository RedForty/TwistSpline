"""
Maya-free reference for the *positions* of the TwistSpline bezier tangent
handles -- a faithful port of ``twistMultiTangentNode`` (see
``maya/tangent_multi_node.py`` / ``src/twistMultiTangentNode.cpp``) restricted to
the open-spline handle geometry that the native rig needs.

Only the handle *positions* matter for the curve shape; the twist basis the C++
node also emits is handled by the RMF elsewhere, so it is omitted here.

A handle's manual ("user") position is expressed as the tangent control's offset
in the CV's local frame -- which, because the control is a direct child of the CV
in the rig, is just the control's local translation. So everything below is plain
vector math (no 4x4 matrices needed), and

    handle_world[i] = cv[i] + ( auto_tan[i] * auto + user_offset[i] * (1 - auto) )
    auto_tan[i]     = smoothTan[i] * smooth + linearTan[i] * (1 - smooth)

matches the node's ``buildDoneTangents`` exactly.
"""

from . import _vmath as vm


def _n(v):
    ln = vm.length(v)
    return [0.0, 0.0, 0.0] if ln < 1e-12 else [v[0] / ln, v[1] / ln, v[2] / ln]


def _vert_tangent(next_len, pre_len, next_nrm, pre_nrm):
    """Normalized curve tangent direction at a vertex (cpp:406-444)."""
    if pre_len == 0.0:
        return [0.0, 1.0, 0.0] if next_len == 0.0 else list(next_nrm)
    if next_len == 0.0:
        return vm.neg(pre_nrm)
    d = vm.dot(pre_nrm, next_nrm)
    if d > 0.999999999:
        y = [1.0, 0.0, 0.0] if abs(pre_nrm[1]) > 0.999999999 else [0.0, 1.0, 0.0]
        return _n(vm.cross(pre_nrm, y))
    if d < -0.999999999:
        return vm.neg(pre_nrm)
    binv = _n(vm.cross(pre_nrm, next_nrm))
    return _n(vm.add(vm.cross(binv, pre_nrm), vm.cross(binv, next_nrm)))


def multi_tangent_handles(cvs, in_w=None, out_w=None, in_s=None, out_s=None,
                          in_a=None, out_a=None, in_user=None, out_user=None,
                          start_tension=2.0):
    """Return (in_handles, out_handles): world positions of the bezier handles.

    in_handles[0] and out_handles[-1] are unused (None) -- the first CV has no
    in-handle and the last no out-handle. Each per-CV list defaults to the
    "all auto, all smooth, unit weight, no manual offset" case.
    """
    n = len(cvs)

    def _d(x, v):
        return [v] * n if x is None else list(x)

    in_w, out_w = _d(in_w, 1.0), _d(out_w, 1.0)
    in_s, out_s = _d(in_s, 1.0), _d(out_s, 1.0)
    in_a, out_a = _d(in_a, 1.0), _d(out_a, 1.0)
    in_user = [[0.0, 0.0, 0.0]] * n if in_user is None else [list(v) for v in in_user]
    out_user = [[0.0, 0.0, 0.0]] * n if out_user is None else [list(v) for v in out_user]

    # legs: in-leg points to prev CV, out-leg to next CV
    in_leglen = [0.0] * n
    out_leglen = [0.0] * n
    in_norm = [[0.0, 0.0, 0.0] for _ in range(n)]
    out_norm = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(1, n):
        leg = vm.sub(cvs[i - 1], cvs[i])
        ln = vm.length(leg)
        in_leglen[i] = ln
        in_norm[i] = vm.scale(leg, 1.0 / ln) if ln else [0.0, 0.0, 0.0]
        out_leglen[i - 1] = ln
        out_norm[i - 1] = vm.neg(in_norm[i])
    in_leglen[0] = 0.0
    out_leglen[n - 1] = 0.0

    in_smooth = [[0.0, 0.0, 0.0] for _ in range(n)]
    out_smooth = [[0.0, 0.0, 0.0] for _ in range(n)]
    # interior smooth handles
    for i in range(1, n - 1):
        tan = _vert_tangent(out_leglen[i], in_leglen[i], out_norm[i], in_norm[i])
        in_smooth[i] = vm.scale(vm.neg(tan), in_leglen[i] * in_w[i] / 3.0)
        out_smooth[i] = vm.scale(tan, out_leglen[i] * out_w[i] / 3.0)
    # endpoints (open): extrapolate from the neighbour, startTension at BOTH ends
    out_smooth[0] = vm.scale(
        vm.sub(vm.add(cvs[1], vm.scale(in_smooth[1], start_tension)), cvs[0]),
        out_w[0] / 2.0)
    in_smooth[n - 1] = vm.scale(
        vm.sub(vm.add(cvs[n - 2], vm.scale(out_smooth[n - 2], start_tension)), cvs[n - 1]),
        in_w[n - 1] / 2.0)

    in_linear = [[0.0, 0.0, 0.0] for _ in range(n)]
    out_linear = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(1, n - 1):
        in_len = in_leglen[i] * in_w[i] / 3.0
        out_len = out_leglen[i] * out_w[i] / 3.0
        in_dir = _n(vm.sub(vm.add(cvs[i - 1], vm.scale(out_smooth[i - 1], out_s[i - 1])), cvs[i]))
        out_dir = _n(vm.sub(vm.add(cvs[i + 1], vm.scale(in_smooth[i + 1], in_s[i + 1])), cvs[i]))
        in_linear[i] = vm.scale(in_dir, in_len)
        out_linear[i] = vm.scale(out_dir, out_len)
    out_linear[0] = vm.scale(
        vm.sub(vm.add(cvs[1], vm.scale(in_smooth[1], in_s[1])), cvs[0]),
        1.0 / (3.0 - in_s[1]))
    in_linear[n - 1] = vm.scale(
        vm.sub(vm.add(cvs[n - 2], vm.scale(out_smooth[n - 2], out_s[n - 2])), cvs[n - 1]),
        1.0 / (3.0 - out_s[n - 2]))

    in_handles = [None] * n
    out_handles = [None] * n
    for i in range(n):
        in_auto = vm.add(vm.scale(in_smooth[i], in_s[i]), vm.scale(in_linear[i], 1.0 - in_s[i]))
        out_auto = vm.add(vm.scale(out_smooth[i], out_s[i]), vm.scale(out_linear[i], 1.0 - out_s[i]))
        in_done = vm.add(vm.scale(in_auto, in_a[i]), vm.scale(in_user[i], 1.0 - in_a[i]))
        out_done = vm.add(vm.scale(out_auto, out_a[i]), vm.scale(out_user[i], 1.0 - out_a[i]))
        if i > 0:
            in_handles[i] = vm.add(cvs[i], in_done)
        if i < n - 1:
            out_handles[i] = vm.add(cvs[i], out_done)
    return in_handles, out_handles
