"""
Maya-free core of the rider constraint -- the per-parameter pipeline and frame
sampling from ``riderConstraint.cpp::compute`` (lines 558-601).

This deliberately stops at *geometry*: it returns the sampled :class:`Frame`
for every (spline, param) pair plus the normalized weights. The Maya node layer
turns those frames into oriented transforms using Maya's own
``MMatrix``/``MQuaternion``/``MEulerRotation`` so the rotation math is identical
to the C++ node. Keeping the param math here makes it testable headlessly.
"""

import math


def solve_rider(splines, weights, spline_lens, end_params, params,
                global_offset=0.0, global_spread=1.0, scale_compensation=1.0,
                use_cycle=False, normalize=1.0, norm_value=1.0,
                use_global_min=False, min_global_param=0.0,
                use_global_max=False, max_global_param=1.0):
    """Sample every (spline, param) pair after the C++ param pipeline.

    splines      : list of core.TwistSpline (already filtered to weight > 0)
    weights      : matching raw weights (normalized internally, summing to 1)
    spline_lens  : per-spline ``splineLength`` input (0 -> derive from spline)
    end_params   : per-spline ``endParam`` input
    params       : list of (param, use_min, min_param, use_max, max_param),
                   indexed by logical param index (gaps filled by the caller)

    Returns (weights, frames, twisted) where ``frames[sIdx][pIdx]`` is a
    :class:`twistspline.core.Frame` and ``twisted`` is True for a single spline.
    """
    n = len(splines)
    w = list(weights)
    s = sum(w)
    if s != 0.0:
        w = [x / s for x in w]

    # Both normValue and globalSpread scale up with the global rig scale.
    g_spread = global_spread * scale_compensation
    g_offset = global_offset * scale_compensation
    norm_val = norm_value * scale_compensation
    do_norm = normalize

    # Single spline bakes twist into the frame; multi weights raw frames and
    # applies twist afterwards (handled in the node).
    twisted = (n == 1)

    frames = []
    for s_idx in range(n):
        spline = splines[s_idx]

        mp = end_params[s_idx]
        mrmp = spline_lens[s_idx]
        if mrmp == 0.0:
            mp = spline.lock_positions[-1]
            mrmp = spline.remap[-1]

        row = []
        for (p_val, use_min, min_param, use_max, max_param) in params:
            p = p_val * g_spread
            p += g_offset

            if use_global_min:
                p = max(min_global_param, p)
            if use_min:
                p = max(min_param, p)
            if use_global_max:
                p = min(max_global_param, p)
            if use_max:
                p = min(max_param, p)

            if do_norm > 0.0:
                p = do_norm * (p * mp / norm_val) + (1.0 - do_norm) * p

            if use_cycle:
                if p >= 0.0:
                    p = math.fmod(p, mrmp)
                else:
                    p = mrmp - math.fmod(-p, mrmp)

            row.append(spline.matrix_at_param(p, twisted))
        frames.append(row)

    return w, frames, twisted
