# TwistSpline — Pure-Python port (Phase 0)

A dependency-free Python reimplementation of the TwistSpline kernel, built
*alongside* the original C++ (none of the C++ is touched). This is **Phase 0**:
the math core plus the tools to verify it — including a parity harness that
checks it produces identical poses to the compiled Maya plugin.

## Why pure Python (no numpy)?

`core.py` is a faithful, readable port of `src/twistSpline.h` using the *Maya*
operator semantics from `src/twistSpline_maya.h` (standard dot/cross/quaternion
rotation — the generic template helpers in `twistSplineUtils.h` are dead code in
the compiled plugin). It uses only the standard library, so it runs in **any**
interpreter, including Maya's bundled Python, with nothing to install.
numpy/vectorization is a later performance phase; this layer is the correctness
reference.

## Layout

```
python/
  twistspline/
    _vmath.py    # 3-vector / quaternion helpers (port of twistSpline_maya.h)
    solve.py     # tridiagonal solver + twist/param solves (port of solve* methods)
    core.py      # the kernel: TwistSpline / TwistSplineSegment (port of twistSpline.h)
  tests/
    test_core.py # Rungs 1-2: self-consistency asserts (no Maya, no C++)
  viz_maya.py    # Rung 3: draw the frames in the Maya viewport
  maya_parity.py # Rung 4: compare against the compiled C++ plugin / dump golden JSON
```

## The verification ladder

### Rung 1-2 — self-consistency (run anywhere)

No Maya, no C++ required:

```bash
python python/tests/test_core.py        # any Python 3
# or:
mayapy python/tests/test_core.py        # Maya's bundled interpreter
```

Checks every sampled frame is orthonormal, never flips, has monotonic
arc-length, is a pure RMF (zero twist) by default, distributes user twist evenly
by arc-length, and honors end-CV orientation.

### Rung 3 — see it in the Maya viewport

```python
import sys; sys.path.append("/path/to/TwistSpline/python")
import viz_maya
viz_maya.demo_pure()      # even, flip-free RMF (no twist)
viz_maya.demo_twist()     # a full turn of user twist, distributed by length
```

Draws the curve plus an RGB axis tripod (tan=red, norm=green, binorm=blue) at
each sample so you can watch the even, pop-free twist.

### Rung 4 — parity against the compiled C++ plugin

Build a rig with the existing `twistSplineBuilder`, then:

```python
import sys; sys.path.append("/path/to/TwistSpline/python")
import maya_parity
# Live comparison: reads the C++ node inputs, rebuilds in Python, samples at the
# rider's params, and reports max position / orientation error vs the rider.
maya_parity.compare("twistSpline1", "riderConstraint1")
# Or capture a repeatable regression file:
maya_parity.dump_golden("twistSpline1", "riderConstraint1", "/tmp/golden.json")
```

`compare` reconstructs the spline from the node exactly as
`twistSplineNode.cpp` does, replicates the rider's parameter transform, and
compares orientations as quaternions (not Euler angles, which differ by
rotate-order and ±360° wraps even when identical).

## Quick API example

```python
from twistspline.core import make_spline
spline = make_spline([[0,0,0], [3,2,1], [6,0,3], [9,-2,1], [12,0,0]])
frame = spline.matrix_at_param(3.0)   # -> Frame(tan, norm, binorm, tran, scl, twist)
```

## Status / next phases

- **Phase 0 (this):** kernel + verification. ✅
- Phase 1: `MPxData` + `twistSpline` locator node + draw override.
- Phase 2: `riderConstraint` node.
- Phase 3: auto-tangent node + builder + AE templates.
- Phase 4: parity polish (pinning/stretch, multi-spline blend, scale, closed, cycle).
- Phase 5: performance (caching, parallel-eval, optional Numba on the RMF walk).
