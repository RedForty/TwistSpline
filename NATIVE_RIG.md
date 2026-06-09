# All-Native-Node TwistSpline Rig

A build-time tool that emits a working TwistSpline rig made of **100% stock Maya
nodes** — no `TwistSpline` plugin, nothing in `scripts/` required at animation
time. The delivered scene has zero dependencies; the animator just opens it.

It reproduces the real (C++) TwistSpline feature-for-feature and has been
validated **directly against the compiled C++ plugin**, joint-for-joint, to
sub-millimetre position and sub-0.1° orientation for all smooth-tangent work.

- Unified entry point: `python/twistspline/maya/rig_builder.py` (native/cpp/python)
- Builder / verifiers: `python/twistspline/maya/native_builder.py`
- Maya-free references: `native_ref.py`, `tangent_ref.py`, `solve.py`, `core.py`
- C++ parity harness: `python/twistspline/maya/compare_rigs.py`

---

## Quick start

```python
from twistspline.maya import native_builder

cvs = [[0,0,0],[3,2,1],[6,0,3],[9,-2,1],[12,0,0]]
rig = native_builder.build_native_spline(cvs, num_joints=10)   # the rig
native_builder.finalize_rig(rig)                               # shapes/colors/cleanup

native_builder.verify_frames(rig)      # vs the Maya-free reference
native_builder.verify_tangents(rig)    # tangent handles vs the reference
```

`build_native_spline(cv_positions, num_joints, spread=3.0, name="nativeTS",
samples_per_interval=20, pins=None, orient_cvs=None, tan_rest=None)` returns a
dict with `cv_ctrls`, `twist_ctrl`, `out_ctrl`/`in_ctrl` (+ buffers), `joints`,
`curve`, `up_curve`, etc.

### Controls (match the real rig)

| Control | Channels | Meaning |
|---|---|---|
| CV control | translate/rotate, `UseOrient`, `Pin`, `PinParam` | the hull vertex; `Pin` locks its param, `UseOrient` injects its orientation |
| Twist control (child of CV) | `rotateX`, `UseTwist` | injects twist at that CV; `UseTwist` is the twist lock weight |
| In/Out tangent control | translate, `Auto`, `Smooth`, `Weight` | bezier handle; `Auto` blends auto↔manual, `Smooth` smooth↔linear, `Weight` handle length |

`Pin` and `UseTwist` are **live 0..1 blends** (animatable without a rebuild).

---

## Unified builder — one control rig, three backends

`rig_builder.build(...)` is the single entry point for all three TwistSpline
backends. They build from identical inputs and return the **same normalized dict**,
and — critically — they all drive the **same production control rig**
(`builder.mkTwistSplineControllers`: CV / twist / tangent / master controls), so a
control-rig improvement is inherited by every backend.

```python
from twistspline.maya import rig_builder

cvs = [[0,0,0],[3,2,1],[6,0,3],[9,-2,1],[12,0,0]]
rig = rig_builder.build("L_tail", cv_positions=cvs, num_joints=10,
                        rig_type="native")     # or "cpp" / "python"
```

| `rig_type` | Backend | Plugin required |
|---|---|---|
| `"native"` | 100% stock Maya nodes (this document) | none |
| `"cpp"` | compiled `TwistSpline` plugin | `TwistSpline` |
| `"python"` | `pyTwistSplinePlugin` scripted nodes | `pyTwistSpline` |

Returns: `rig_type, group, cv_ctrls, twist_ctrl, out_ctrl, in_ctrl, joints, curve,
master`. The native backend wires its node graph onto the shared controls via
`build_native_spline(..., controls=<adapted dict>)`; standalone
`build_native_spline()` (locator controls) still exists as a lightweight,
plugin-free path used by the verifiers and the parity suite.

The native curve is fed **world-space** CV positions, so its curve/joints group
stays at world origin — the master drives the rig through the *controls*, exactly
like the C++/python backends (never parent that group under the master, or it
double-transforms). On the shared-control path, call `hide_guts(rig)` (the shipping
rig keeps its already-shaped production controls; `finalize_rig` is for the
standalone locator rig).

---

## How it works (node graph)

1. **Curve** — a live degree-3 bezier-form NURBS curve whose control points are
   driven by the CV controls and the tangent handles.
2. **Tangents** — a native port of `twistMultiTangent`: half-angle Catmull-Rom
   smooth tangents (leg-length × weight / 3), smooth↔linear blend, and a control
   that rides an auto-driven buffer (so it tracks the computed tangent in Auto
   mode); the spline follows the control's world position, so a manual offset
   always bends it.
3. **Orientation (RMF)** — a **double-reflection** rotation-minimizing frame
   (the same method the C++ kernel uses), transported along the curve and baked
   into a second "upCurve" so each joint reads its frame by sampling.
4. **Twist** — `solveTwistParamMatrix` solved live as a native Thomas-algorithm
   node chain: twist-pinned CVs hold their value, unpinned CVs float between
   pins (exact for fractional `UseTwist`).
5. **UseOrient** — the control-up residual at orient-pinned CVs, distributed by
   arc length.
6. **Pin** — `solveParamMatrix` solved live as a native Thomas chain; joints
   redistribute along the curve as pinned controls move.
7. **Joints** — distributed by **arc length** (like the C++ rider) via
   `motionPath`, with the curve param recovered (`nearestPointOnCurve`) to sample
   the upCurve for orientation.

### The key insight

The C++ kernel's twist / orientation / parameter behaviours are tridiagonal
linear solves (`src/twistSpline.h`). Those don't map onto stock nodes directly —
**but each one reduces** to a short, exact native form:

- The **0/1-pin** cases collapse to piecewise interpolation between pinned knots.
- The **fractional** cases are the full tridiagonal solve, which — because every
  diagonal is −1 — runs as a compact **Thomas-algorithm node chain** (verified
  exact vs the kernel solvers over thousands of random fractional cases).

So there is no runtime solver: the solves are unrolled into a static node graph.

---

## Validation

Two layers, both reproducible:

- **`verify_frames` / `verify_tangents`** — compare the live rig to Maya-free
  references (`native_ref`, `tangent_ref`) that are faithful ports of the kernel.
- **`compare_rigs.parity_suite()`** — builds the real **C++** rig and the native
  rig from identical inputs and compares **joint-for-joint** across a systematic
  battery (CV moves, orient, the full twist range incl. fractional `UseTwist`,
  Pin incl. fractional, tangent Weight/Smooth/Auto + moves, combined poses).
  `parity_suite(use_production=True)` runs the same battery against the native rig
  built through `rig_builder` (wired onto the **shared** production controls, the
  shipping path) — it matches the locator run exactly, confirming the shared-control
  native graph is numerically identical to the validated one.

Result: **sub-millimetre position, <0.06° orientation** for all smooth-tangent
states. Several real bugs were found and fixed via this parity work — e.g. a
non-unit `angleBetween` transport axis, and the RMF method itself (switched from
`angleBetween` parallel transport to **double-reflection** to match the kernel).

### Known limitation — kinked (non-G1) tangents

When a tangent is made **non-smooth** at a CV (`Auto` < 1, a hard-yanked handle,
or `Smooth` = 0), the RMF normal has a genuine *discontinuity* at that CV. The
continuous node-graph upCurve represents that discontinuity slightly differently
from the C++ deformer's per-segment LUT, so joints right at such a kink differ by
~0.5–4° (it is **systematic, not a sampling/resolution issue**). All *smooth*
tangent work is exact. Closing this would require rebuilding the orientation path
per-segment with discontinuities at kinked CVs.

---

## Cosmetics

`finalize_rig(rig)` is optional and purely cosmetic (it never touches the math):
colored NURBS control shapes, internal guts (upCurve + arcLengthDimension nodes)
hidden, non-animatable channels locked/hidden, and a `…_controls` selection set.
Use it on the **standalone** locator rig.

`hide_guts(rig)` is the guts-hiding step on its own (upCurve + arcLengthDimension
indicators into a hidden group) without re-shaping controls — use it for the
**shared-control** rig, whose production controls are already shaped.
`rig_builder.build("...", rig_type="native")` calls it for you.
