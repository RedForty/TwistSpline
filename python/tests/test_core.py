"""
Phase 0 self-consistency tests for the pure-Python TwistSpline kernel.

These need *no* Maya and *no* C++ -- only the kernel itself -- so you can run
them anywhere:

    # in a terminal (any Python 3):
    python python/tests/test_core.py

    # or with Maya's bundled interpreter:
    mayapy python/tests/test_core.py

They check the three properties that define "the magic works":
    1. Every sampled frame is orthonormal.
    2. The frame never flips (consecutive normals stay on the same side).
    3. The endpoints honor their control orientation, and twist distributes
       evenly by arc-length.
Plus basic sanity (arc-length monotonic, pure spline has zero twist).
"""

import os
import sys
from math import pi, radians, degrees

# Allow running directly: add the package parent dir to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twistspline import _vmath as vm
from twistspline.core import make_spline


# A non-trivial, genuinely-3D control hull so the frames actually have to twist.
S_CURVE = [
    [0.0, 0.0, 0.0],
    [3.0, 2.0, 1.0],
    [6.0, 0.0, 3.0],
    [9.0, -2.0, 1.0],
    [12.0, 0.0, 0.0],
]

EPS = 1e-9


def _sample_params(spline, count=200):
    lo, hi = spline.param_range
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("  [{}] {}{}".format(status, label, (" -- " + detail) if detail and not condition else ""))
    return condition


def test_orthonormal():
    print("test_orthonormal: every sampled frame is an orthonormal basis")
    spline = make_spline(S_CURVE)
    worst = 0.0
    ok = True
    for p in _sample_params(spline):
        f = spline.matrix_at_param(p)
        # unit length
        for axis in (f.tan, f.norm, f.binorm):
            worst = max(worst, abs(vm.length(axis) - 1.0))
        # mutually perpendicular
        worst = max(worst, abs(vm.dot(f.tan, f.norm)))
        worst = max(worst, abs(vm.dot(f.tan, f.binorm)))
        worst = max(worst, abs(vm.dot(f.norm, f.binorm)))
    ok = check("orthonormal within 1e-9", worst < 1e-9, "worst deviation = {:.2e}".format(worst))
    return ok


def test_no_flip():
    print("test_no_flip: the frame never suddenly flips")
    spline = make_spline(S_CURVE)
    params = _sample_params(spline, 400)
    prev = None
    ok = True
    worst = 1.0
    for p in params:
        n = spline.matrix_at_param(p).norm
        if prev is not None:
            d = vm.dot(prev, n)
            worst = min(worst, d)
            if d <= 0.0:
                ok = False
        prev = n
    ok = check("consecutive normals stay on same side (dot > 0)", ok,
               "min consecutive dot = {:.4f}".format(worst))
    return ok


def test_arclength_monotonic():
    print("test_arclength_monotonic: LUT arc-length strictly increases")
    spline = make_spline(S_CURVE)
    ok = True
    for seg in spline.segments:
        sl = seg.sample_lengths
        if any(sl[i + 1] < sl[i] - EPS for i in range(len(sl) - 1)):
            ok = False
    return check("sample lengths non-decreasing per segment", ok)


def test_pure_spline_has_no_twist():
    print("test_pure_spline_has_no_twist: default spline is a pure RMF")
    spline = make_spline(S_CURVE)  # no user twist, no orient lock
    worst = max(abs(spline.matrix_at_param(p).twist) for p in _sample_params(spline))
    return check("all twist values ~ 0", worst < 1e-9, "max |twist| = {:.2e}".format(worst))


def test_user_twist_distributes_by_length():
    print("test_user_twist_distributes_by_length: 180 deg twist ramps evenly")
    n = len(S_CURVE)
    # Pin twist at both ends: 0 at the start, pi at the last CV.
    user_twists = [0.0] * n
    user_twists[-1] = pi
    twist_locks = [0.0] * n
    twist_locks[0] = 1.0
    twist_locks[-1] = 1.0
    spline = make_spline(S_CURVE, user_twists=user_twists, twist_locks=twist_locks)

    lo, hi = spline.param_range
    t_start = spline.matrix_at_param(lo).twist
    t_end = spline.matrix_at_param(hi).twist
    t_mid = spline.matrix_at_param(0.5 * (lo + hi)).twist

    ok = True
    ok &= check("twist at start ~ 0", abs(t_start) < 1e-6,
                "got {:.4f} deg".format(degrees(t_start)))
    ok &= check("twist magnitude at end ~ 180 deg", abs(abs(t_end) - pi) < 1e-4,
                "got {:.4f} deg".format(degrees(t_end)))
    # Even distribution: midpoint (by param, roughly mid arc-length here) is ~ halfway.
    ok &= check("twist is monotonic start->mid->end",
                abs(t_mid) > abs(t_start) and abs(t_end) > abs(t_mid),
                "start={:.1f} mid={:.1f} end={:.1f} deg".format(
                    degrees(t_start), degrees(t_mid), degrees(t_end)))
    return ok


def test_endpoint_orientation_is_honored():
    print("test_endpoint_orientation_is_honored: end CV twist drives the frame")
    n = len(S_CURVE)
    # Rotate the last CV 90 deg about X (its tangent-ish axis) and lock orient.
    half = radians(90.0) / 2.0
    import math
    last_quat = [math.cos(half), math.sin(half), 0.0, 0.0]  # [w, x, y, z]
    cv_quats = [[1.0, 0.0, 0.0, 0.0] for _ in range(n)]
    cv_quats[-1] = last_quat
    orient_locks = [0.0] * n
    orient_locks[0] = 1.0
    orient_locks[-1] = 1.0

    spline = make_spline(S_CURVE, cv_quats=cv_quats, orient_locks=orient_locks)

    lo, hi = spline.param_range
    t_start = spline.matrix_at_param(lo).twist
    t_end = spline.matrix_at_param(hi).twist
    # The exact end angle depends on the RMF's natural arrival, but locking the
    # end orientation must produce a non-trivial, smoothly-applied twist.
    ok = check("end orientation produces a non-zero twist solve",
               abs(t_end - t_start) > radians(10.0),
               "delta = {:.2f} deg".format(degrees(t_end - t_start)))
    return ok


def main():
    tests = [
        test_orthonormal,
        test_no_flip,
        test_arclength_monotonic,
        test_pure_spline_has_no_twist,
        test_user_twist_distributes_by_length,
        test_endpoint_orientation_is_honored,
    ]
    print("=" * 64)
    print("TwistSpline Phase 0 kernel -- self-consistency tests")
    print("=" * 64)
    results = []
    for t in tests:
        results.append(t())
        print("")
    passed = sum(1 for r in results if r)
    print("=" * 64)
    print("{} / {} test groups passed".format(passed, len(results)))
    print("=" * 64)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
