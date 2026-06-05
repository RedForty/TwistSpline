"""
Maya-free math for ``twistTangent`` (port of ``twistTangentNode.cpp::compute``).

Three independent outputs, matching the node's three compute branches:

* :func:`smooth_tangent`  -> the auto smooth-tangent vector, the re-orthonormalized
  up/normal, and the (tan, nrm, bin) basis for the twist matrix.
* :func:`linear_target`   -> where the *paired* tangent should aim in linear mode.
* :func:`out_tangent`     -> the final tangent position (world), blending the free
  (manual) tangent, the smooth tangent, and the linear target by ``smooth``/``auto``.

All vectors are plain ``[x, y, z]`` lists in world space. The node layer pulls
translations / the Y axis out of the input matrices and applies
``parentInverseMatrix`` to the two outputs that need it (out, outTwistMat).
"""

from . import _vmath as vm


def _n(v):
    """Safe normalize: zero-length -> zero vector, like ``MVector::normal()``."""
    ln = vm.length(v)
    if ln < 1e-12:
        return [0.0, 0.0, 0.0]
    return [v[0] / ln, v[1] / ln, v[2] / ln]


def smooth_tangent(pre_pos, cur_pos, next_pos, cur_y, weight, backpoint, endpoint):
    """Return (smooth_tan, nrm, tan, bin) -- the C++ aSmoothTan branch.

    ``cur_y`` is the current vertex matrix's Y axis (row 1), used as the
    up-vector seed exactly as ``curTMat.asMatrix()[1]`` is in the C++.
    """
    pre_leg = vm.sub(pre_pos, cur_pos)
    next_leg = vm.sub(next_pos, cur_pos)
    pre_len = vm.length(pre_leg)
    next_len = vm.length(next_leg)

    pre_norm = _n(pre_leg)
    post_norm = _n(next_leg)
    dot = vm.dot(pre_norm, post_norm)

    if abs(dot) >= 0.999999999 or pre_len == 0.0:  # linear
        tan = _n(next_leg)
        nrm = cur_y
        binv = _n(vm.cross(nrm, tan))
        nrm = _n(vm.cross(tan, binv))
        smo = vm.scale(next_leg, 1.0 / 3.0)
    elif not endpoint:  # nonlinear
        binv = _n(vm.cross(pre_norm, post_norm))
        tan = _n(vm.add(vm.cross(binv, pre_norm), vm.cross(binv, post_norm)))
        smo = vm.scale(tan, next_len / 3.0)
        nrm = cur_y
        binv = _n(vm.cross(nrm, tan))
        nrm = _n(vm.cross(tan, binv))
    else:  # endpoint twist definition
        tan = post_norm
        binv = _n(vm.cross(pre_norm, post_norm))
        smo = _n(vm.add(vm.cross(binv, pre_norm), vm.cross(binv, post_norm)))
        smo = vm.scale(smo, next_len / 3.0)
        nrm = cur_y
        binv = _n(vm.cross(nrm, tan))
        nrm = _n(vm.cross(tan, binv))

    if backpoint:
        tan = vm.neg(tan)
        binv = vm.neg(binv)

    smo = vm.scale(smo, weight)
    return smo, nrm, tan, binv


def linear_target(smooth_tan, smooth, cur_pos):
    """C++ aOutLinearTarget branch: smooth_tan * smooth + cur_pos (world)."""
    return vm.add(vm.scale(smooth_tan, smooth), cur_pos)


def out_tangent(in_pos, cur_pos, smooth_tan, next_linear_target, smooth, auto):
    """C++ aOut branch: blended tangent position in *world* space.

    The node multiplies the result by parentInverseMatrix afterwards.
    """
    lin = vm.scale(_n(vm.sub(next_linear_target, cur_pos)), vm.length(smooth_tan))
    result = vm.add(vm.scale(lin, 1.0 - smooth), vm.scale(smooth_tan, smooth))
    free_leg = vm.sub(in_pos, cur_pos)
    result = vm.add(vm.scale(free_leg, 1.0 - auto), vm.scale(result, auto))
    return vm.add(result, cur_pos)
