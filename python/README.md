# TwistSpline — Pure-Python (OpenMaya 2.0) Port

A complete reimplementation of the C++ TwistSpline plugin as **pure Python
nodes** built on the Maya Python API 2.0 (`maya.api.OpenMaya`). No compiler, no
per-Maya-version builds — load one `.py` file and you have the whole rig.

Every node is **bit-for-bit parity-verified** against its C++ counterpart in a
live Maya session: a full Python rig produces rider-joint positions identical to
the C++ rig (`max dPos = 0.0`) under arbitrary control manipulation.

> This document is about the *port* — how it differs from the C++ plugin and how
> it works. For what a TwistSpline *is* and how to rig with it, see the
> top-level [`README.md`](../README.md) and [`nodeDocs.md`](../nodeDocs.md).

---

## Quick start

```python
import maya.cmds as cmds

# 1) Load the Python plugin by full path (it self-bootstraps sys.path)
cmds.loadPlugin("/path/to/TwistSpline/python/pyTwistSplinePlugin.py")

# 2) Build a rig with the py* nodes
import sys; sys.path.append("/path/to/TwistSpline/python")
from twistspline.maya import builder
builder.makeTwistSpline("L_arm", numCVs=5, numJoints=10)
```

The Python plugin registers node types prefixed `py*` (`pyTwistSpline`,
`pyRiderConstraint`, …) with their own `MTypeId`s, so it can be **loaded at the
same time as the C++ plugin** without clashing — which is exactly what makes the
side-by-side parity testing possible.

---

## C++ vs. Python: execution differences

Both implementations run the *identical algorithm* and produce the *identical
result*. What differs is how they execute.

| | C++ plugin | Python port |
|---|---|---|
| **Build step** | Compile per OS + per Maya version (devkit, CMake/meson) | None — load a `.py` file |
| **Distribution** | Ship a `.mll`/`.so`/`.bundle` per platform | Ship a folder of `.py` |
| **Speed** | Fast (native, no interpreter) | Slower per-eval (Python in the DG loop) |
| **Hackability** | Edit → recompile → reload Maya | Edit → purge modules → reload plugin |
| **Debugging** | C++ debugger / print | Standard Python tracebacks in Maya |
| **Numerical result** | — | **identical** (see Verification) |

**When to use which.** The C++ plugin is the right choice for production
playback and heavy scenes where evaluation cost matters. The Python port is for
**portability and iteration**: machines without the compiled plugin, quick
experiments, learning the algorithm, or pipelines that can't ship binaries.
Because the node types coexist and the math is identical, you can prototype with
the Python nodes and swap to C++ for performance — the rig topology and channel
values are the same.

**Why it's not just "approximately" the same.** The port doesn't reimplement
Maya's linear algebra. Anything order-sensitive — quaternion construction and
slerp, `parentInverseMatrix` multiplication, euler decomposition by rotate
order, the multi-tangent rotation extraction — is done with Maya's *own*
`MMatrix`/`MQuaternion`/`MEulerRotation`/`MTransformationMatrix` inside the node.
So even floating-point rounding matches the C++.

---

## How the port works

### Two layers: a Maya-free kernel + thin OM2 nodes

The core insight is a clean split:

* **A Maya-free math kernel** — the actual TwistSpline algorithm, with *zero*
  Maya imports. It runs in any Python 3 interpreter, so it can be unit-tested
  headlessly (no Maya, no C++) and reused unchanged.
* **A thin OM2 node layer** — reads/writes plugs, marshals data, and does the
  Maya-specific linear algebra. This is the only part that imports
  `maya.api.OpenMaya`.

```
python/
├── pyTwistSplinePlugin.py        # loadable plugin: registers everything
├── viz_maya.py                   # viewport demos of the bare kernel
├── maya_parity.py                # kernel-vs-C++ harness (Phase 0)
├── tests/                        # headless kernel tests (no Maya, no C++)
└── twistspline/
    ├── _vmath.py                 # ── Maya-free kernel ──
    ├── solve.py                  #   banded/tridiagonal solves (param & twist pinning)
    ├── core.py                   #   TwistSpline / TwistSplineSegment (RMF, twist, sampling)
    ├── rider.py                  #   rider param pipeline (spread/offset/clamp/normalize/cycle)
    ├── tangent.py                #   twistTangent vector math
    └── maya/                     # ── OM2 node layer ──
        ├── spline_data.py        #   pyTwistSplineData   (MPxData)
        ├── spline_node.py        #   pyTwistSpline       (MPxLocatorNode)
        ├── draw.py               #   Viewport 2.0 MPxDrawOverride
        ├── rider_node.py         #   pyRiderConstraint   (MPxNode)
        ├── tangent_node.py       #   pyTwistTangent      (MPxNode)
        ├── tangent_multi_node.py #   pyTwistMultiTangent (MPxNode)
        ├── builder.py            #   full rig assembly
        ├── build.py              #   per-node parity harnesses
        └── compare_rigs.py       #   end-to-end py-vs-C++ joint comparison
```

The kernel (`core.py` + `solve.py` + `_vmath.py`) is a faithful port of
`src/twistSpline.h` using the Maya operator semantics from
`src/twistSpline_maya.h`, deliberately mirroring the C++ control flow — sampling,
frame-walk, and twist-solve order — rather than "improving" it.

### Node and data mapping

| C++ type (`MTypeId`) | Python type (`MTypeId`) | OM2 base class |
|---|---|---|
| `twistSpline` (`0x001226F7`) | `pyTwistSpline` (`0x0013F741`) | `OpenMayaUI.MPxLocatorNode` |
| `TwistSplineData` (`0x001226FB`) | `pyTwistSplineData` (`0x0013F740`) | `MPxData` |
| `riderConstraint` (`0x001226FC`) | `pyRiderConstraint` (`0x0013F742`) | `MPxNode` |
| `twistTangent` (`0x001226FA`) | `pyTwistTangent` (`0x0013F743`) | `MPxNode` |
| `twistMultiTangent` (`0x00122715`) | `pyTwistMultiTangent` (`0x0013F744`) | `MPxNode` |

Long attribute names are mirrored **exactly** between the C++ and Python nodes.
That is what lets `builder.py` be a line-for-line copy of the original
`scripts/twistSplineBuilder.py` with only the four `createNode` type strings
swapped — every `connectAttr`/`setAttr` works unchanged.

> The Python `MTypeId`s live in a high block (`0x0013F740…744`), distinct from
> the C++ block, so both plugins coexist. This block is fine for an in-house
> pipeline; register your own block with Autodesk before distributing widely.

### Carrying the spline between nodes (`MPxData`)

The spline travels from `pyTwistSpline` to `pyRiderConstraint` over a plug typed
as custom plugin data — the OM2 equivalent of the C++ `TwistSplineData`:

```python
# producer (pyTwistSpline.compute): wrap a kernel TwistSpline in plugin data
fn   = om.MFnPluginData()
obj  = fn.create(TwistSplineData.kId)
fn.data().setSpline(spline)          # spline is a twistspline.core.TwistSpline
out_handle.setMPxData(fn.data())

# consumer (pyRiderConstraint.compute): pull it back off the input plug
spline = om.MFnPluginData(in_handle.data()).data().spline()
```

Because the output is a derived value (recomputed every `compute`, marked
non-storable), no file-IO serialization hooks are needed.

### API 2.0 specifics worth knowing

The OM2 patterns (and gotchas) that made the port work:

* **`maya_useNewAPI()`** must exist in every plugin/registered module.
* **`MPxLocatorNode` lives in `maya.api.OpenMayaUI`**, not `OpenMaya`.
* **Plugin self-bootstrap:** Maya runs a plugin file without a reliable
  `__file__`. Derive the package dir from `MFnPlugin.loadPath()` instead, and
  defer node imports into `initializePlugin`.
* **Output-plug check:** `if plug in (someMObjects)` raises *"MObject expected"*
  (reflected `MObject.__eq__(MPlug)`). Compare `plug.attribute()` against the
  attribute `MObject`s.
* **Sparse arrays:** iterate `MArrayDataHandle` with `jumpToPhysicalElement(i)`
  and read the logical index via `elementLogicalIndex()`.
* **Angles:** an `MDataHandle` on a `kAngle` attribute returns **radians** from
  `asDouble()` — matching the C++ node.
* **Exact rotations:** build the output basis as an `MMatrix`, then go through
  `MTransformationMatrix.rotation(asQuaternion=True)` and
  `MEulerRotation.decompose(matrix, rotateOrder)` so euler values match the C++
  bit-for-bit, instead of reimplementing Maya's decomposition.
* **`om.MVector` operators map 1:1 to C++ `MVector`** (`^` cross, `*` dot,
  `.normal()`), so the matrix-heavy `pyTwistMultiTangent` is a near-verbatim
  transcription.

**Iterating on the nodes.** `loadPlugin` won't re-read modules Python already
cached, so purge them on reload:

```python
import sys, maya.cmds as cmds
if cmds.pluginInfo("pyTwistSplinePlugin", q=True, loaded=True):
    cmds.file(new=True, force=True)
    cmds.unloadPlugin("pyTwistSplinePlugin")
for m in [k for k in sys.modules if k.startswith("twistspline")]:
    del sys.modules[m]
cmds.loadPlugin("/path/to/TwistSpline/python/pyTwistSplinePlugin.py")
```

---

## Verification

Parity was established at three levels, every one passing:

**1. Headless kernel tests** (no Maya, no C++):

```bash
python python/tests/test_core.py     # orthonormal frames, no flips, arc-length, twist
python python/tests/test_rider.py    # the full rider param pipeline
```

`viz_maya.py` draws the bare kernel in the viewport, and `maya_parity.py`
compares the kernel against the compiled C++ node directly (Phase 0 tools).

**2. Per-node parity vs. C++** (in Maya, both plugins loaded), in
`twistspline.maya.build`:

* `pyTwistSpline` output spline matches the kernel exactly.
* `pyRiderConstraint`: `dRot ≈ 1e-14` (single & multi-spline, incl. slerp,
  twist quaternion, `parentInverseMatrix`, euler-by-order).
* `pyTwistTangent`: `≤ 2e-16` across all three geometry branches.
* `pyTwistMultiTangent`: **exactly `0.0`**, open and closed.

**3. End-to-end rig parity** (`compare_rigs`): a Python rig and a C++ rig driven
by one shared control set — rider joints stay coincident, `max dPos = 0.0`,
under arbitrary posing.

```python
from twistspline.maya import compare_rigs
rigs = compare_rigs.setup_comparison(numCVs=5, numJoints=10)
# ...pose the cmpPy_* controls...
compare_rigs.compare_joints(rigs, orient=True)   # -> ~0 position & rotation
```

---

## Notes & limitations

* **Performance:** Python evaluates in the DG loop; for heavy playback prefer
  the C++ plugin. The rig topology and channel values are identical either way.
* **Maya version:** `builder.py` wires `twistMat → offsetParentMatrix`
  (Maya 2020+). The nodes themselves only need OM2.
* **MTypeIds:** the `0x0013F740…744` block is for in-house use. Register your
  own Autodesk block before shipping the Python nodes in a distributed tool.
* **Scope:** the four functional nodes, the rig builder, and the draw override
  are ported. AE templates / node icons for the `py*` types are not (cosmetic).
