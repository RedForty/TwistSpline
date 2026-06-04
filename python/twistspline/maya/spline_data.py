"""
Custom plugin data that carries a TwistSpline between Maya nodes.

This is the Python/OM2 equivalent of the C++ ``TwistSplineData`` (MPxData) in
``src/twistSplineData.h``. It is what travels over the ``outputSpline`` plug
from the spline node to (eventually) the rider node.

Because the spline is *derived* data -- it is recomputed from the node's inputs
on every ``compute`` and the output plug is marked non-storable -- we do NOT
implement the file-IO serialization hooks (readASCII/writeASCII/...). If we
later want the data to survive scene save without a recompute, those can be
added; for now the recompute path is authoritative and matches how the C++
node's output behaves in practice.
"""

import maya.api.OpenMaya as om

from ..core import TwistSpline


def maya_useNewAPI():
    pass


class TwistSplineData(om.MPxData):
    # A high MTypeId block, deliberately distinct from the C++ plugin's ids so
    # the Python and C++ plugins can be loaded at the same time. For a shipping
    # pipeline, register your own block with Autodesk.
    kId = om.MTypeId(0x0013F740)
    kName = "pyTwistSplineData"

    def __init__(self):
        super(TwistSplineData, self).__init__()
        self._spline = None  # a twistspline.core.TwistSpline (or None)

    # -- our accessors -----------------------------------------------------
    def setSpline(self, spline):
        self._spline = spline

    def spline(self):
        return self._spline

    # -- MPxData overrides -------------------------------------------------
    def copy(self, src):
        # ``src`` is another TwistSplineData. The spline is immutable after the
        # producing node's compute (it is rebuilt wholesale each time), so it is
        # safe to share the reference rather than deep-copy.
        self._spline = src.spline() if isinstance(src, TwistSplineData) else None

    def typeId(self):
        return TwistSplineData.kId

    def name(self):
        return TwistSplineData.kName

    @staticmethod
    def creator():
        return TwistSplineData()
