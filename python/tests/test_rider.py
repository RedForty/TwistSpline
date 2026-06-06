"""
Headless tests for the Maya-free rider param pipeline (twistspline.rider).

These validate the per-parameter arithmetic ported from
``riderConstraint.cpp::compute`` -- spread/offset, the four min/max clamps in
order, the normalize remap, and cycling -- by recomputing the expected param
independently and confirming ``solve_rider`` sampled the spline at exactly that
param. No Maya, no C++.

    python python/tests/test_rider.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twistspline import _vmath as vm
from twistspline.core import make_spline
from twistspline.rider import solve_rider

S_CURVE = [
    [0.0, 0.0, 0.0], [3.0, 2.0, 1.0], [6.0, 0.0, 3.0],
    [9.0, -2.0, 1.0], [12.0, 0.0, 0.0],
]
EPS = 1e-9


def check(label, condition, detail=""):
    print("  [{}] {}{}".format("PASS" if condition else "FAIL", label,
                                (" -- " + detail) if detail and not condition else ""))
    return condition


def _expected_p(p_val, spline, *, g_off, g_spread, sclcmp, do_norm, norm_value,
                use_cycle, use_gmin, gmin, use_gmax, gmax,
                use_min, pmin, use_max, pmax):
    """Reference reimplementation of the C++ pipeline for one param."""
    g_spread *= sclcmp
    g_off *= sclcmp
    norm_val = norm_value * sclcmp
    mp = spline.lock_positions[-1]
    mrmp = spline.remap[-1]

    p = p_val * g_spread + g_off
    if use_gmin: p = max(gmin, p)
    if use_min:  p = max(pmin, p)
    if use_gmax: p = min(gmax, p)
    if use_max:  p = min(pmax, p)
    if do_norm > 0.0:
        p = do_norm * (p * mp / norm_val) + (1.0 - do_norm) * p
    if use_cycle:
        if p >= 0.0:
            p = math.fmod(p, mrmp)
        else:
            p = mrmp - math.fmod(-p, mrmp)
    return p


def _frame_matches(frame, spline, p, twisted):
    ref = spline.matrix_at_param(p, twisted)
    return (vm.length(vm.sub(frame.tran, ref.tran)) < EPS and
            vm.length(vm.sub(frame.norm, ref.norm)) < EPS and
            vm.length(vm.sub(frame.binorm, ref.binorm)) < EPS and
            abs(frame.twist - ref.twist) < EPS)


def _run_case(label, **kw):
    spline = make_spline(S_CURVE)
    param_inputs = [
        (0.0, kw["use_min"], kw["pmin"], kw["use_max"], kw["pmax"]),
        (0.25, kw["use_min"], kw["pmin"], kw["use_max"], kw["pmax"]),
        (0.5, kw["use_min"], kw["pmin"], kw["use_max"], kw["pmax"]),
        (1.0, kw["use_min"], kw["pmin"], kw["use_max"], kw["pmax"]),
        (1.7, kw["use_min"], kw["pmin"], kw["use_max"], kw["pmax"]),
    ]
    weights, frames, twisted = solve_rider(
        [spline], [1.0], [0.0], [1.0], param_inputs,
        global_offset=kw["g_off"], global_spread=kw["g_spread"],
        scale_compensation=kw["sclcmp"], use_cycle=kw["use_cycle"],
        normalize=kw["do_norm"], norm_value=kw["norm_value"],
        use_global_min=kw["use_gmin"], min_global_param=kw["gmin"],
        use_global_max=kw["use_gmax"], max_global_param=kw["gmax"])

    ok = (abs(weights[0] - 1.0) < EPS and twisted)
    for j, (p_val, *_rest) in enumerate(param_inputs):
        exp = _expected_p(p_val, spline, g_off=kw["g_off"], g_spread=kw["g_spread"],
                          sclcmp=kw["sclcmp"], do_norm=kw["do_norm"],
                          norm_value=kw["norm_value"], use_cycle=kw["use_cycle"],
                          use_gmin=kw["use_gmin"], gmin=kw["gmin"],
                          use_gmax=kw["use_gmax"], gmax=kw["gmax"],
                          use_min=kw["use_min"], pmin=kw["pmin"],
                          use_max=kw["use_max"], pmax=kw["pmax"])
        ok = ok and _frame_matches(frames[0][j], spline, exp, twisted)
    return check(label, ok)


_BASE = dict(g_off=0.0, g_spread=1.0, sclcmp=1.0, do_norm=1.0, norm_value=1.0,
             use_cycle=False, use_gmin=False, gmin=0.0, use_gmax=False, gmax=1.0,
             use_min=False, pmin=0.0, use_max=False, pmax=1.0)


def _case(**over):
    d = dict(_BASE)
    d.update(over)
    return d


def main():
    print("rider param-pipeline tests")
    results = []
    results.append(_run_case("default normalize", **_case()))
    results.append(_run_case("no normalize (raw param)", **_case(do_norm=0.0)))
    results.append(_run_case("global spread+offset", **_case(g_spread=2.0, g_off=1.5)))
    results.append(_run_case("scale compensation", **_case(sclcmp=2.5, g_spread=1.3, g_off=0.7)))
    results.append(_run_case("global min clamp", **_case(do_norm=0.0, use_gmin=True, gmin=4.0)))
    results.append(_run_case("global max clamp", **_case(do_norm=0.0, use_gmax=True, gmax=8.0)))
    results.append(_run_case("per-param min+max", **_case(do_norm=0.0, use_min=True, pmin=2.0,
                                                          use_max=True, pmax=9.0)))
    results.append(_run_case("cycle", **_case(do_norm=0.0, use_cycle=True, g_spread=8.0)))
    results.append(_run_case("normValue remap", **_case(norm_value=2.0)))

    passed = sum(1 for r in results if r)
    print("\n{}/{} rider pipeline tests passed".format(passed, len(results)))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
