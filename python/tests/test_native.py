"""
Validate the native-node reference (twistspline.native_ref) against the kernel.

Confirms the all-native scheme (forward RMF + piecewise-linear twist pinning +
useOrient end residual) reproduces the kernel's twisted frame for the real rig
configurations -- to within the RMF discretization floor (set by samples/seg).

    python python/tests/test_native.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twistspline import _vmath as vm
from twistspline.core import make_spline
from twistspline.native_ref import native_frames

S_CURVE = [[0, 0, 0], [3, 2, 1], [6, 0, 3], [9, -2, 1], [12, 0, 0]]


def _quat_axis_angle(axis, ang):
    axis = vm.normalized(axis)
    s = math.sin(ang / 2.0)
    return [math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]


def _max_frame_err(cfg_label, cv_quats=None, twist_vals=None, twist_locks=None,
                   orient_locks=None, samples_per_seg=16):
    ref = make_spline(S_CURVE, cv_quats=cv_quats, user_twists=twist_vals,
                      twist_locks=twist_locks, orient_locks=orient_locks)
    lo, hi = ref.param_range
    params = [lo + (hi - lo) * i / 120 for i in range(121)]

    nat = native_frames(S_CURVE, params, cv_quats=cv_quats, twist_vals=twist_vals,
                        twist_locks=twist_locks, orient_locks=orient_locks,
                        samples_per_seg=samples_per_seg)

    max_pos = max_nrm = 0.0
    for p, fr in zip(params, nat):
        k = ref.matrix_at_param(p, twisted=True)
        max_pos = max(max_pos, vm.length(vm.sub(fr["pos"], k.tran)))
        d = max(-1.0, min(1.0, vm.dot(fr["normal"], k.norm)))
        max_nrm = max(max_nrm, math.degrees(math.acos(d)))
    print("  {:<34} max dPos={:.2e}  max dFrame={:6.3f} deg".format(
        cfg_label, max_pos, max_nrm))
    return max_nrm


def main():
    print("native_ref vs kernel (16 samples/seg):")
    end_tan = make_spline(S_CURVE).matrix_at_param(make_spline(S_CURVE).param_range[1],
                                                   twisted=False).tan
    rotated_end = [[1, 0, 0, 0]] * 5
    rotated_end[4] = _quat_axis_angle(end_tan, math.radians(50.0))

    results = []
    # 1. pure RMF (default rig: no twist, no orient pins)
    results.append(_max_frame_err("pure RMF"))
    # 2. user twist pinned at CV0 and CV3
    results.append(_max_frame_err("twist pins CV0,CV3",
                                  twist_vals=[0, 0, 0, math.radians(40), 0],
                                  twist_locks=[1, 0, 0, 1, 0]))
    # 3. useOrient on first + last, rotated end control (your rig)
    results.append(_max_frame_err("useOrient first+last",
                                  cv_quats=rotated_end,
                                  orient_locks=[1, 0, 0, 0, 1]))
    # 4. everything together
    results.append(_max_frame_err("orient first+last + twist pin",
                                  cv_quats=rotated_end,
                                  twist_vals=[0, 0, 0, math.radians(40), 0],
                                  twist_locks=[1, 0, 0, 1, 0],
                                  orient_locks=[1, 0, 0, 0, 1]))

    print("\nconvergence (config 4) vs samples/seg:")
    for s in (2, 4, 8, 16, 32):
        _max_frame_err("  samples/seg = {}".format(s), cv_quats=rotated_end,
                       twist_vals=[0, 0, 0, math.radians(40), 0],
                       twist_locks=[1, 0, 0, 1, 0], orient_locks=[1, 0, 0, 0, 1],
                       samples_per_seg=s)

    worst = max(results)
    ok = worst < 0.2  # within the RMF discretization floor at 16 samples/seg
    print("\nworst frame error at 16 samples/seg: {:.3f} deg -> {}".format(
        worst, "OK" if ok else "REVIEW"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
