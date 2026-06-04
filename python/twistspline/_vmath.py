"""
Minimal 3-vector / quaternion math used by the TwistSpline reference kernel.

This is a faithful port of the *Maya* operator set in ``src/twistSpline_maya.h``
(NOT the generic template helpers in ``src/twistSplineUtils.h``, which are dead
code in the compiled plugin). That means:

    dot       -> standard dot product            (MVector::operator*)
    cross     -> standard right-handed cross      (MVector::operator^)
    normalized-> a / |a|                          (MVector::normal)
    reject    -> perpendicular component, normed  (twistSpline_maya.h reject)
    rotateBy  -> standard quaternion rotation      (MVector::rotateBy)

Vectors are plain ``list``/``tuple`` of three floats. Quaternions are
``[w, x, y, z]`` (scalar first). Everything is pure-Python stdlib so the kernel
runs in any interpreter -- including Maya's bundled Python -- with no
third-party dependencies. numpy vectorization is a later (performance) concern.
"""

from math import sqrt


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(a):
    return sqrt(dot(a, a))


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def neg(a):
    return [-a[0], -a[1], -a[2]]


def normalized(a):
    return scale(a, 1.0 / length(a))


def cross(a, b):
    """Standard right-handed cross product (matches Maya's ``a ^ b``)."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def reject(onto, n):
    """The component of ``n`` perpendicular to ``onto``, normalized.

    Mirror of ``reject`` in twistSpline_maya.h:
        (n - ((n . onto) / (onto . onto)) * onto).normal()
    """
    d = dot(n, onto) / dot(onto, onto)
    return normalized(sub(n, scale(onto, d)))


def rotate_by_quat(v, q):
    """Rotate vector ``v`` by unit quaternion ``q = [w, x, y, z]``.

    Standard right-handed Hamilton rotation, equivalent to Maya's
    ``MVector::rotateBy(MQuaternion)``. Implemented via the well-known
    ``v + 2w(u x v) + 2(u x (u x v))`` form where ``u`` is the vector part.
    """
    u = [q[1], q[2], q[3]]
    t = scale(cross(u, v), 2.0)
    return add(v, add(scale(t, q[0]), cross(u, t)))


def basis_to_quat(x_axis, y_axis, z_axis):
    """Convert an orthonormal basis (as the rows X/Y/Z of a rotation matrix)
    into a quaternion ``[w, x, y, z]``.

    The TwistSpline rider builds its output matrix with rows
    (tan, norm, binorm) -> (X, Y, Z), so pass those three vectors here to get
    the orientation quaternion for parity comparisons.
    """
    m00, m01, m02 = x_axis
    m10, m11, m12 = y_axis
    m20, m21, m22 = z_axis
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = [w, x, y, z]
    n = sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def quat_angle_between(qa, qb):
    """Smallest rotation angle (radians) between two unit quaternions.

    Sign/double-cover safe -- treats q and -q as identical -- which is what you
    want when comparing orientations for parity.
    """
    d = abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3])
    d = max(min(d, 1.0), -1.0)
    from math import acos
    return 2.0 * acos(d)
