"""
Pure-Python reference implementation of the TwistSpline kernel.

This is a faithful, dependency-free port of the C++ header ``src/twistSpline.h``
(the ``TwistSpline`` / ``TwistSplineSegment`` classes) using the *Maya* operator
semantics from ``src/twistSpline_maya.h``. The goal is bit-for-bit *behavioral*
parity with the compiled Maya plugin, so this module deliberately mirrors the
C++ control flow -- including its sampling, frame-walk and twist-solve order --
rather than "improving" it.

Pipeline (matches the C++ exactly):
    1. Each cubic Bezier segment is sampled into a lookup table (LUT) of
       points + tangents.                                  -> compute_spline_points
    2. A Rotation-Minimizing Frame is walked across the LUT via the
       Double Reflection Method (Wang et al. 2008).        -> double_reflect
    3. Segments are chained, each seeded by reflecting the
       previous segment's last frame across the joint.     -> build_segments
    4. The residual angle between the RMF and each CV's actual orientation is
       measured (post_angle), distributed by arc-length with a tridiagonal
       solve + euler filter, and applied around the tangent. -> solve_twist
    5. Sampling a parameter remaps length -> param and returns a frame.

No numpy: vectors are plain lists of floats (see ``_vmath``). Vectorization is
a later performance concern; this layer is the readable, verifiable reference.
"""

import bisect
from collections import namedtuple
from math import sqrt, sin, cos, acos, pi

from . import _vmath as vm
from .solve import solve_param_matrix, solve_twist_param_matrix


TAU = 2.0 * pi

# The frame returned when sampling the spline at a parameter.
#   tan, norm, binorm : orthonormal axes (X, Y, Z of the output matrix)
#   tran              : position
#   scl               : scale
#   twist             : the interpolated twist angle at this parameter
Frame = namedtuple("Frame", ["tan", "norm", "binorm", "tran", "scl", "twist"])


def linear_index(t, samp):
    """Port of the C++ ``linearIndex`` free function.

    Find which segment of the sorted sample vector ``samp`` the value ``t``
    falls in. Returns ``(seg_t, seg_idx)`` where ``seg_t`` is the 0..1 position
    inside segment ``seg_idx`` (can fall outside 0..1 when ``t`` is out of
    range, matching the C++ extrapolation behavior).
    """
    n = len(samp)
    ub = bisect.bisect_right(samp, t)  # std::upper_bound

    if ub == 0:
        seg_idx = 0
        seg_t = (t / samp[1]) if n > 1 else 0.0
    elif ub == n:
        seg_idx = n - 2
        if n > 1:
            seg_t = (t - samp[seg_idx]) / (samp[seg_idx + 1] - samp[seg_idx])
        else:
            seg_t = 1.0
    else:
        lb = ub - 1
        seg_t = (t - samp[lb]) / (samp[ub] - samp[lb])
        seg_idx = lb
        # Snap values very close to the end to the next segment (de-jitter)
        if seg_idx < n - 2 and seg_t > 0.99999999999:
            seg_t = 0.0
            seg_idx += 1
    return seg_t, seg_idx


def _bezier_point(pts, t):
    """Cubic Bezier point at ``t`` (Bernstein form). Port of ``compute``."""
    if t <= 0.0:
        return list(pts[0])
    if t >= 1.0:
        return list(pts[3])
    mt = 1.0 - t
    a = mt * mt * mt
    b = mt * mt * t * 3.0
    c = mt * t * t * 3.0
    d = t * t * t
    return [
        a * pts[0][k] + b * pts[1][k] + c * pts[2][k] + d * pts[3][k]
        for k in range(3)
    ]


def _bezier_tangent(dverts, t):
    """Quadratic Bezier (the derivative curve) at ``t``. Port of ``computeTangent``."""
    if t <= 0.0:
        return list(dverts[0])
    if t >= 1.0:
        return list(dverts[2])
    mt = 1.0 - t
    a = mt * mt
    b = mt * t * 2.0
    c = t * t
    return [
        a * dverts[0][k] + b * dverts[1][k] + c * dverts[2][k]
        for k in range(3)
    ]


def compute_spline_points(pts, lut_steps, want_tangents):
    """Port of ``TwistSplineSegment::computeSplinePoints``.

    Evaluates the cubic Bezier into ``lut_steps + 1`` points (and optionally
    tangents) using the same forward-difference scheme as the C++ so that
    floating-point accumulation matches.
    """
    p0, p1, p2, p3 = pts
    a = [p3[k] - 3 * p2[k] + 3 * p1[k] - p0[k] for k in range(3)]
    b = [3 * p2[k] - 6 * p1[k] + 3 * p0[k] for k in range(3)]
    c = [3 * p1[k] - 3 * p0[k] for k in range(3)]

    h = 1.0 / lut_steps
    h2 = h * h
    h3 = h2 * h
    point_steps = lut_steps + 1

    p_out = [None] * point_steps
    p_out[0] = list(p0)
    d = list(p0)
    fd = [a[k] * h3 + b[k] * h2 + c[k] * h for k in range(3)]
    fd2 = [6 * a[k] * h3 + 2 * b[k] * h2 for k in range(3)]
    fd3 = [6 * a[k] * h3 for k in range(3)]
    for i in range(1, point_steps):
        d = [d[k] + fd[k] for k in range(3)]
        fd = [fd[k] + fd2[k] for k in range(3)]
        fd2 = [fd2[k] + fd3[k] for k in range(3)]
        p_out[i] = list(d)

    t_out = None
    if want_tangents:
        t_out = [None] * point_steps
        tan = [3 * (p1[k] - p0[k]) for k in range(3)]
        td = [3 * a[k] * h2 + 2 * b[k] * h for k in range(3)]
        td2 = [6 * a[k] * h2 for k in range(3)]
        t_out[0] = vm.normalized(tan)
        for i in range(1, point_steps):
            tan = [tan[k] + td[k] for k in range(3)]
            td = [td[k] + td2[k] for k in range(3)]
            t_out[i] = vm.normalized(tan)

    return p_out, t_out


def double_reflect(i_norm, points, tangents, lut_steps):
    """Port of ``TwistSplineSegment::doubleReflect`` -- the RMF walk.

    Carries one initial up-vector along the curve using two reflections per
    step (Wang et al., "Computation of Rotation Minimizing Frames"). Returns
    ``(normals, binormals, sample_lengths, units)``.
    """
    n = lut_steps + 1
    normals = [None] * n
    binormals = [None] * n
    sample_lengths = [0.0] * n
    units = [None] * lut_steps

    normals[0] = vm.reject(tangents[0], i_norm)
    binormals[0] = vm.cross(tangents[0], normals[0])

    for i in range(lut_steps):
        v1 = vm.sub(points[i + 1], points[i])
        v12 = vm.dot(v1, v1)
        c1 = 2.0 / v12
        t_li = vm.sub(tangents[i], vm.scale(v1, c1 * vm.dot(v1, tangents[i])))
        v2 = vm.sub(tangents[i + 1], t_li)
        dv2 = vm.dot(v2, v2)
        if abs(dv2) < 1.0e-15:
            normals[i + 1] = normals[i]
            binormals[i + 1] = binormals[i]
            continue
        c2 = 2.0 / dv2

        ln = sqrt(v12)
        sample_lengths[i + 1] = sample_lengths[i] + ln
        units[i] = vm.scale(v1, 1.0 / ln)

        r_li = vm.sub(normals[i], vm.scale(v1, c1 * vm.dot(v1, normals[i])))
        normals[i + 1] = vm.sub(r_li, vm.scale(v2, c2 * vm.dot(v2, r_li)))
        binormals[i + 1] = vm.cross(tangents[i + 1], normals[i + 1])

    return normals, binormals, sample_lengths, units


class TwistSplineSegment(object):
    """One cubic Bezier segment with its LUT and frames. Port of the C++ class."""

    def __init__(self, verts, scl_verts, quats, i_norm, lut_steps):
        # verts/scl_verts: 4 control points each; quats: 4 orientations
        self.verts = verts
        self.scl_verts = scl_verts
        self.quats = quats
        self.i_norm = i_norm
        self.lut_steps = lut_steps

        self._build_dverts()

        self.points, self.tangents = compute_spline_points(verts, lut_steps, True)
        self.scales, _ = compute_spline_points(scl_verts, lut_steps, False)
        (self.rnormals, self.rbinormals,
         self.sample_lengths, self.units) = double_reflect(
            i_norm, self.points, self.tangents, lut_steps)

        # Twisted frames + twist values are filled in later by apply_twist()
        self.tnormals = list(self.rnormals)
        self.tbinormals = list(self.rbinormals)
        self.twist_vals = [0.0] * (lut_steps + 1)

    def _build_dverts(self):
        v = self.verts
        s = self.scl_verts
        self.d1verts = [[3 * (v[i + 1][k] - v[i][k]) for k in range(3)] for i in range(3)]
        self.d2verts = [[2 * (self.d1verts[i + 1][k] - self.d1verts[i][k]) for k in range(3)] for i in range(2)]
        self.s1verts = [[3 * (s[i + 1][k] - s[i][k]) for k in range(3)] for i in range(3)]
        self.s2verts = [[2 * (self.s1verts[i + 1][k] - self.s1verts[i][k]) for k in range(3)] for i in range(2)]

    def get_length(self):
        return self.sample_lengths[-1] if self.sample_lengths else 0.0

    def compute_tran(self, t):
        return _bezier_point(self.verts, t)

    def compute_scale(self, t):
        return _bezier_point(self.scl_verts, t)

    def compute_tran_tangent(self, t):
        return _bezier_tangent(self.d1verts, t)

    def post_angle(self):
        """Angle between the RMF's final normal and the end CV's up-axis.

        Port of ``TwistSplineSegment::postAngle``. This residual is what the
        twist solve distributes so the spline honors each control's orientation.
        """
        y = [0.0, 1.0, 0.0]
        n = vm.rotate_by_quat(y, self.quats[3])
        t_last = self.tangents[self.lut_steps]
        b_last = self.rbinormals[self.lut_steps]
        f_last = self.rnormals[self.lut_steps]
        cv_last = vm.reject(t_last, n)

        dd = vm.dot(cv_last, f_last)
        dd = max(min(dd, 1.0), -1.0)
        angle = acos(dd)
        disc = vm.dot(b_last, cv_last)
        return -angle if disc >= 0 else angle

    def apply_twist(self, start_angle, end_angle):
        """Rotate the raw frames around the tangent, interpolating the twist
        angle by arc-length. Port of ``TwistSplineSegment::applyTwist``.
        """
        start_angle = -start_angle
        end_angle = -end_angle

        n = len(self.rnormals)
        self.tnormals = [None] * n
        self.tbinormals = [None] * n
        self.twist_vals = [0.0] * n

        ln = self.get_length()
        for i in range(self.lut_steps + 1):
            perc = self.sample_lengths[i] / ln
            angle = (end_angle - start_angle) * perc + start_angle
            self.twist_vals[i] = angle
            x = self.rnormals[i]
            y = self.rbinormals[i]
            tn = self.tangents[i]
            self.tnormals[i] = vm.add(vm.scale(x, cos(angle)), vm.scale(y, sin(angle)))
            self.tbinormals[i] = vm.cross(tn, self.tnormals[i])

    def matrix_at_param(self, raw_t, normals, binormals):
        """Frame at ``raw_t`` (0..1 normalized *length* within the segment).

        Port of ``TwistSplineSegment::matrixAtParam`` (the ``#else`` branch that
        the C++ marks "KEEP IT HERE" -- simple interpolation, least jitter).
        """
        ln = self.get_length()
        len_t = raw_t * ln
        t, i = linear_index(len_t, self.sample_lengths)
        # seg_t is the 0..1 *param* value across the whole segment
        seg_t = (t + i) / self.lut_steps

        if abs(t) < 1.0e-11:  # exactly on a sample
            return Frame(
                tan=self.tangents[i], norm=normals[i], binorm=binormals[i],
                tran=self.points[i], scl=self.scales[i], twist=self.twist_vals[i])

        if t < 0.0:  # extrapolate before the start
            dv = self.d1verts[0]
            dv = dv if vm.dot(dv, dv) < 1.0e-11 else vm.normalized(dv)
            sv = self.s1verts[0]
            sv = sv if vm.dot(sv, sv) < 1.0e-11 else vm.normalized(sv)
            return Frame(
                tan=self.tangents[0], norm=normals[0], binorm=binormals[0],
                tran=vm.add(vm.scale(dv, len_t), self.points[0]),
                scl=vm.add(vm.scale(sv, len_t), self.scales[0]),
                twist=self.twist_vals[0])

        if t > 0.99999999999:  # extrapolate past the end
            dv = self.d1verts[2]
            dv = dv if vm.dot(dv, dv) < 1.0e-11 else vm.normalized(dv)
            sv = self.s1verts[2]
            sv = sv if vm.dot(sv, sv) < 1.0e-11 else vm.normalized(sv)
            # NB: C++ reads tbinormals here regardless of raw/twisted mode
            return Frame(
                tan=self.tangents[-1], norm=normals[-1], binorm=self.tbinormals[-1],
                tran=vm.add(vm.scale(dv, len_t - ln), self.points[-1]),
                scl=vm.add(vm.scale(sv, len_t - ln), self.scales[-1]),
                twist=self.twist_vals[-1])

        # Main path: evaluate position/tangent mathematically, interpolate the
        # normal between LUT samples, then rebuild an orthonormal basis.
        tran = self.compute_tran(seg_t)
        scl = self.compute_scale(seg_t)
        tan = vm.normalized(self.compute_tran_tangent(seg_t))

        n1 = normals[i]
        n2 = normals[i + 1]
        nn = [n1[k] * (1.0 - t) + n2[k] * t for k in range(3)]
        binorm = vm.normalized(vm.cross(tan, nn))
        norm = vm.cross(binorm, tan)

        twist = self.twist_vals[i] * (1.0 - t) + self.twist_vals[i + 1] * t
        return Frame(tan=tan, norm=norm, binorm=binorm, tran=tran, scl=scl, twist=twist)


class TwistSpline(object):
    """A chain of TwistSplineSegments. Port of the C++ ``TwistSpline`` class."""

    def __init__(self, lut_steps=20):
        self.lut_steps = lut_steps
        self.segments = []
        self.remap = []
        self.total_length = 0.0
        # stored inputs
        self.verts = []
        self.scales = []
        self.quats = []
        self.lock_positions = []
        self.lock_values = []
        self.user_twists = []
        self.twist_locks = []
        self.orient_locks = []

    # -- construction ------------------------------------------------------

    def set_verts(self, verts, scales, quats, lock_positions, lock_values,
                  user_twists, twist_locks, orient_locks):
        """Set all inputs and build the spline. Port of ``TwistSpline::setVerts``.

        verts/scales : 3*numSegs+1 control points (CVs interleaved with the
                       out/in bezier tangents)
        quats        : one orientation per vert (only the CV ones are used)
        lock_*/...   : one value per CV (numSegs+1 of them)
        """
        self.verts = verts
        self.scales = scales
        self.quats = quats
        self.lock_positions = lock_positions
        self.lock_values = lock_values
        self.user_twists = user_twists
        self.twist_locks = twist_locks
        self.orient_locks = orient_locks

        self.build_segments()
        if not self.segments:
            return

        ulens = [0.0]
        runner = 0.0
        for seg in self.segments:
            runner += seg.get_length()
            ulens.append(runner)
        self.total_length = runner

        self.remap = solve_param_matrix(lock_positions, ulens, lock_values)
        self.solve_twist()

    def build_segments(self):
        """Port of ``TwistSpline::buildSegments`` -- including the cross-segment
        frame continuity (each segment seeded by reflecting the previous
        segment's frame across the joint)."""
        num_verts = len(self.verts)
        self.segments = []
        if num_verts < 2:
            return
        num_segs = (num_verts - 1) // 3

        for i in range(num_segs):
            if i == 0:
                y = [0.0, 1.0, 0.0]
                i_norm = vm.rotate_by_quat(y, self.quats[3 * i])
            else:
                pre = self.segments[i - 1]
                last = pre.lut_steps - 1  # NB: second-to-last sample, as in C++
                d = pre.rnormals[last]
                a = pre.tangents[last]
                b = vm.normalized(vm.sub(self.verts[3 * i + 1], self.verts[3 * i]))
                ab = vm.add(a, b)
                n = b if vm.length(ab) == 0.0 else vm.normalized(ab)
                i_norm = vm.sub(d, vm.scale(n, 2.0 * vm.dot(d, n)))

            vv = [self.verts[3 * i + j] for j in range(4)]
            ss = [self.scales[3 * i + j] for j in range(4)]
            qq = [self.quats[3 * i + j] for j in range(4)]
            self.segments.append(
                TwistSplineSegment(vv, ss, qq, i_norm, self.lut_steps))

    def solve_twist(self):
        """Port of ``TwistSpline::solveTwist``.

        Distributes (a) the residual orientation angle of each CV and (b) the
        user twist values, both arc-length weighted via the tridiagonal solve,
        applies an euler filter, then rotates each segment's frames.
        """
        num_segs = len(self.segments)
        seg_lens = [0.0] * (num_segs + 1)
        orient_vals = [0.0] * (num_segs + 1)
        for i in range(num_segs):
            seg_lens[i + 1] = self.segments[i].get_length() + seg_lens[i]
            orient_vals[i + 1] = -self.segments[i].post_angle()

        ori_map = solve_twist_param_matrix(orient_vals, seg_lens, self.orient_locks)

        # Euler filter so the spline can wind past 360 deg without snapping back
        for i in range(1, len(ori_map)):
            diff = ori_map[i] - ori_map[i - 1]
            sign = 1.0
            if diff < 0.0:
                diff = -diff
                sign = -1.0
            count = 0
            while count < 10:
                if diff < pi:
                    break
                diff -= TAU
                count += 1
            ori_map[i] -= count * sign * TAU

        twist_map = solve_twist_param_matrix(self.user_twists, seg_lens, self.twist_locks)

        for i in range(num_segs):
            self.segments[i].apply_twist(
                ori_map[i] + twist_map[i], ori_map[i + 1] + twist_map[i + 1])

    # -- sampling ----------------------------------------------------------

    def matrix_at_param(self, t, twisted=True):
        """Frame at length-parameter ``t``. Port of ``TwistSpline::matrixAtParam``.

        This is the authoritative sampling path -- it is what the C++ rider
        calls per parameter (the templated ``matricesAtParams`` is unused dead
        code in the plugin).
        """
        seg_t, seg_idx = linear_index(t, self.remap)
        seg = self.segments[seg_idx]
        if twisted:
            return seg.matrix_at_param(seg_t, seg.tnormals, seg.tbinormals)
        return seg.matrix_at_param(seg_t, seg.rnormals, seg.rbinormals)

    def matrices_at_params(self, params, twisted=True):
        return [self.matrix_at_param(p, twisted) for p in params]

    @property
    def param_range(self):
        """The (min, max) length-parameter domain, i.e. the remap endpoints."""
        if not self.remap:
            return (0.0, 0.0)
        return (self.remap[0], self.remap[-1])

    # -- LUT getters (for visualization / debugging) ----------------------

    def get_points(self):
        out = []
        for seg in self.segments:
            out.extend(seg.points)
        return out

    def get_tangents(self):
        out = []
        for seg in self.segments:
            out.extend(seg.tangents)
        return out

    def get_normals(self, twisted=True):
        out = []
        for seg in self.segments:
            out.extend(seg.tnormals if twisted else seg.rnormals)
        return out

    def get_binormals(self, twisted=True):
        out = []
        for seg in self.segments:
            out.extend(seg.tbinormals if twisted else seg.rbinormals)
        return out


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def _well_pose(locks):
    """Force the first lock to 1.0 if every lock is zero, mirroring the
    guards in twistSplineNode.cpp:375-379. Leaves a non-trivial array alone.
    """
    locks = list(locks)
    if locks and not any(v > 0.0 for v in locks):
        locks[0] = 1.0
    return locks


def _catmull_tangents(cvs):
    """Catmull-Rom style auto in/out bezier tangents for a list of CV points.

    Returns (out_tans, in_tans) where out_tans[i] is the outgoing tangent of
    CV i and in_tans[i] is the incoming tangent of CV i.
    """
    n = len(cvs)
    out_tans = [None] * n
    in_tans = [None] * n
    for k in range(n):
        if k == 0:
            m = vm.sub(cvs[1], cvs[0])
        elif k == n - 1:
            m = vm.sub(cvs[n - 1], cvs[n - 2])
        else:
            m = vm.scale(vm.sub(cvs[k + 1], cvs[k - 1]), 0.5)
        out_tans[k] = vm.add(cvs[k], vm.scale(m, 1.0 / 3.0))
        in_tans[k] = vm.sub(cvs[k], vm.scale(m, 1.0 / 3.0))
    return out_tans, in_tans


def make_spline(cv_positions, cv_quats=None, lock_positions=None, lock_values=None,
                user_twists=None, twist_locks=None, orient_locks=None,
                spread=3.0, lut_steps=20, tangents=None):
    """Build a :class:`TwistSpline` from a list of CV positions with sensible
    defaults and automatic (Catmull-Rom) bezier tangents.

    By default the spline is fully position-pinned to evenly spaced params and
    carries no twist -- i.e. a pure RMF, the "even, flip-free by default" case.

    cv_positions : list of [x, y, z]
    cv_quats     : optional list of [w, x, y, z] per CV (default identity)
    spread       : param distance between consecutive CVs when fully locked
    """
    n = len(cv_positions)
    cvs = [list(p) for p in cv_positions]

    if tangents is not None:
        # explicit (e.g. live/edited) bezier handles: in_tans[i] for i>0,
        # out_tans[i] for i<n-1
        in_tans, out_tans = tangents
    else:
        # Default auto bezier tangents, faithful to the real twistMultiTangent node
        # (half-angle Catmull-Rom, leg-length handles, endpoint tension).
        from .tangent_ref import multi_tangent_handles
        in_tans, out_tans = multi_tangent_handles(cvs)

    # Assemble the interleaved vert array: [cv0, out0, in1, cv1, out1, in2, ...]
    verts = []
    for i in range(n - 1):
        verts.append(cvs[i])
        verts.append(out_tans[i])
        verts.append(in_tans[i + 1])
    verts.append(cvs[n - 1])

    scales = [[1.0, 1.0, 1.0] for _ in verts]

    # Per-vert quats; only the CV ones (indices 0, 3, 6, ...) are used by the math.
    if cv_quats is None:
        cv_quats = [[1.0, 0.0, 0.0, 0.0] for _ in range(n)]
    quats = [[1.0, 0.0, 0.0, 0.0] for _ in verts]
    for k in range(n):
        quats[3 * k] = list(cv_quats[k])

    if lock_positions is None:
        lock_positions = [i * spread for i in range(n)]
    if lock_values is None:
        lock_values = [0.0] * n
    if user_twists is None:
        user_twists = [0.0] * n
    if twist_locks is None:
        twist_locks = [0.0] * n
    if orient_locks is None:
        orient_locks = [0.0] * n

    # Well-posing guards. The twist/orientation/parameter solves are singular
    # if *nothing* is pinned (the value is a free constant), so the real rig
    # always pins at least the first CV:
    #   - param  : twistSplineBuilder sets cvs[0].Pin = 1.0
    #              + twistSplineNode.cpp:375 forces lockVals[0] = 1 if none set
    #   - orient : twistSplineNode.cpp:378 forces orientLock[0] = 1 if none set
    #   - twist  : twistSplineBuilder sets tws[0].UseTwist = 1.0
    # We reproduce that here so the convenience builder never hands the kernel a
    # singular system.
    lock_values = _well_pose(lock_values)
    twist_locks = _well_pose(twist_locks)
    orient_locks = _well_pose(orient_locks)

    spline = TwistSpline(lut_steps=lut_steps)
    spline.set_verts(verts, scales, quats, lock_positions, lock_values,
                     user_twists, twist_locks, orient_locks)
    return spline
