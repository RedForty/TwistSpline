"""
Maya-free reference for the native-node TwistSpline reconstruction.

This is the *spec* the all-native Maya rig will implement node-for-node. It uses
only operations that map to stock Maya nodes:

  * curve position / tangent      -> motionPath / pointOnCurveInfo on a live curve
  * forward parallel-transport RMF -> angleBetween / axisAngleToQuat / composeMatrix
  * per-CV twist pinning           -> piecewise-linear-by-arclength (proven exact)
  * useOrient end residual         -> angleBetween at the curve, linearly distributed

No global linear solve. The only Maya-specific thing it *doesn't* do here is read
the curve geometry from a real nurbs/bezier curve -- it borrows position/tangent
from the validated kernel, since a Maya degree-3 curve through the same control
points is the identical cubic.

`native_frames` returns one dict per joint param:
    {pos, tan, normal, binormal, twist}
"""

import math

from .core import make_spline, _well_pose
from . import _vmath as vm

# Sign conventions resolved empirically against the kernel (see test_native.py).
_SIGN = 1.0


def _safe_norm(v):
    ln = vm.length(v)
    return [0.0, 0.0, 0.0] if ln < 1e-12 else [c / ln for c in v]


def _reproject(n, t):
    return _safe_norm(vm.sub(n, vm.scale(t, vm.dot(n, t))))


def _rodrigues(v, axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    return vm.add(vm.add(vm.scale(v, c), vm.scale(vm.cross(axis, v), s)),
                  vm.scale(axis, vm.dot(axis, v) * (1.0 - c)))


def _transport(n, t_prev, t_cur):
    """Minimal-rotation parallel transport — exactly what angleBetween+rotate do."""
    axis = vm.cross(t_prev, t_cur)
    s = vm.length(axis)
    c = vm.dot(t_prev, t_cur)
    if s > 1e-12:
        n = _rodrigues(n, vm.scale(axis, 1.0 / s), math.atan2(s, c))
    return _reproject(n, t_cur)


def _signed_angle(a, b, axis):
    """Signed angle from a to b measured around `axis` (both ⟂ axis)."""
    return math.atan2(vm.dot(vm.cross(a, b), axis), vm.dot(a, b))


def _pw_linear_between_locks(values, arclens, locks):
    """Piecewise-linear-by-arclength between locked knots; flat past the ends.

    This is the proven native-equivalent of solve_twist_param_matrix.
    """
    n = len(values)
    pinned = [i for i, l in enumerate(locks) if l >= 0.5]
    if not pinned:
        return [0.0] * n
    out = [0.0] * n
    for i in range(n):
        if i <= pinned[0]:
            out[i] = values[pinned[0]]
        elif i >= pinned[-1]:
            out[i] = values[pinned[-1]]
        else:
            lo = max(p for p in pinned if p <= i)
            hi = min(p for p in pinned if p >= i)
            if lo == hi:
                out[i] = values[i]
            else:
                f = (arclens[i] - arclens[lo]) / (arclens[hi] - arclens[lo])
                out[i] = values[lo] * (1.0 - f) + values[hi] * f
    return out


def native_frames(cv_positions, joint_params, cv_quats=None, twist_vals=None,
                  twist_locks=None, orient_locks=None, spread=3.0, samples_per_seg=16,
                  tangents=None):
    n = len(cv_positions)
    if cv_quats is None:
        cv_quats = [[1.0, 0.0, 0.0, 0.0]] * n

    # Geometry only (a real Maya degree-3 curve is the identical cubic).
    spline = make_spline(cv_positions, cv_quats=cv_quats, spread=spread, tangents=tangents)
    lo, hi = spline.param_range
    num_segs = len(spline.segments)

    def tan_at(p):
        return spline.matrix_at_param(p, twisted=False).tan

    def pos_at(p):
        return spline.matrix_at_param(p, twisted=False).tran

    # CV knot params and cumulative arc length at each CV.
    cv_param = [spline.remap[i] for i in range(num_segs + 1)]
    cv_arclen = [0.0] * (num_segs + 1)
    for i in range(num_segs):
        cv_arclen[i + 1] = cv_arclen[i] + spline.segments[i].get_length()

    # RMF base = the spline's own double-reflection normals (the method the C++
    # kernel uses); we only sample arc length here for the twist/orient distribution.
    npts = samples_per_seg * num_segs
    sp = [lo + (hi - lo) * i / npts for i in range(npts + 1)]
    pts = [pos_at(p) for p in sp]
    arc = [0.0]
    for i in range(1, len(sp)):
        arc.append(arc[-1] + vm.length(vm.sub(pts[i], pts[i - 1])))

    def _bracket(p):
        for i in range(len(sp) - 1):
            if sp[i] <= p <= sp[i + 1]:
                f = (p - sp[i]) / (sp[i + 1] - sp[i]) if sp[i + 1] > sp[i] else 0.0
                return i, f
        return len(sp) - 2, 1.0

    def rmf_at(p):
        return spline.matrix_at_param(p, twisted=False).norm

    def arclen_at(p):
        i, f = _bracket(p)
        return arc[i] * (1.0 - f) + arc[i + 1] * f

    # Locks, well-posed exactly like the kernel.
    twist_vals = list(twist_vals) if twist_vals else [0.0] * n
    twist_locks = _well_pose(list(twist_locks) if twist_locks else [0.0] * n)
    orient_locks = _well_pose(list(orient_locks) if orient_locks else [0.0] * n)

    # Orientation residual at each CV: angle(RMF -> control up) around the tangent.
    resid = [0.0] * n
    for i in range(n):
        p = cv_param[i]
        t = tan_at(p)
        up = _reproject(vm.rotate_by_quat([0.0, 1.0, 0.0], cv_quats[i]), t)
        resid[i] = _signed_angle(rmf_at(p), up, t)
    resid[0] = 0.0  # the RMF starts anchored here

    ori_dist = _pw_linear_between_locks(resid, cv_arclen, orient_locks)
    tw_dist = _pw_linear_between_locks(twist_vals, cv_arclen, twist_locks)
    total = [ori_dist[i] + tw_dist[i] for i in range(n)]

    def total_at(a):
        for k in range(num_segs):
            if cv_arclen[k] <= a <= cv_arclen[k + 1]:
                f = ((a - cv_arclen[k]) / (cv_arclen[k + 1] - cv_arclen[k])
                     if cv_arclen[k + 1] > cv_arclen[k] else 0.0)
                return total[k] * (1.0 - f) + total[k + 1] * f
        return total[-1]

    out = []
    for p in joint_params:
        t = tan_at(p)
        ang = total_at(arclen_at(p))
        n_ = _reproject(_rodrigues(rmf_at(p), t, _SIGN * ang), t)
        out.append({
            "pos": pos_at(p), "tan": t, "normal": n_,
            "binormal": _safe_norm(vm.cross(t, n_)), "twist": ang,
        })
    return out


def params_at_fractions(cv_positions, fractions, spread=3.0, lut=400):
    """Map arc-length fractions (0..1) to kernel params, for matched sampling.

    Lets a motionPath-driven rig (which works in arc-length fraction) be compared
    against :func:`native_frames` at the same physical points on the curve.
    """
    spline = make_spline(cv_positions, spread=spread)
    lo, hi = spline.param_range
    sp = [lo + (hi - lo) * i / lut for i in range(lut + 1)]
    pts = [spline.matrix_at_param(p, twisted=False).tran for p in sp]
    arc = [0.0]
    for i in range(1, len(sp)):
        arc.append(arc[-1] + vm.length(vm.sub(pts[i], pts[i - 1])))
    total = arc[-1]
    out = []
    for fr in fractions:
        target = fr * total
        for i in range(len(arc) - 1):
            if arc[i] <= target <= arc[i + 1]:
                f = (target - arc[i]) / (arc[i + 1] - arc[i]) if arc[i + 1] > arc[i] else 0.0
                out.append(sp[i] * (1.0 - f) + sp[i + 1] * f)
                break
        else:
            out.append(hi)
    return out
