"""
Linear solves used by the TwistSpline kernel.

Faithful ports of ``solveTridiagonalMatrix``, ``solveParamMatrix`` and
``solveTwistParamMatrix`` from ``src/twistSpline.h``. These are what let any
subset of control vertices "pin" a parameter (position) or twist value while
the rest interpolate smoothly between them.

Each matrix row is stored as ``[sub, diag, super]`` (the three non-zero
diagonals), exactly like the C++ ``std::array<Float, 3>``.
"""


def solve_tridiagonal(mat, res):
    """Thomas-algorithm solve of a tridiagonal system, in linear time.

    Port of ``TwistSpline::solveTridiagonalMatrix``. Operates on copies so the
    inputs are left untouched; returns the solution vector.
    """
    n = len(res)
    mat = [list(row) for row in mat]
    res = list(res)

    # First-row normalization
    mat[0][2] = mat[0][2] / mat[0][1]
    res[0] = res[0] / mat[0][1]

    # Forward sweep
    for i in range(1, n):
        denom = mat[i][1] - mat[i][0] * mat[i - 1][2]
        mat[i][2] = mat[i][2] / denom
        res[i] = (res[i] - mat[i][0] * res[i - 1]) / denom

    # Back substitution
    for i in range(n - 2, -1, -1):
        res[i] = res[i] - mat[i][2] * res[i + 1]

    return res


def solve_param_matrix(rest_vals, current_vals, lock_vals):
    """Port of ``TwistSpline::solveParamMatrix``.

    Solves for the remapped parameter at each CV given:
        rest_vals    -- the parameter each CV gets when fully locked
        current_vals -- the cumulative arc-lengths at each CV
        lock_vals    -- 0..1 weight of how locked each CV's parameter is
    """
    rv, cv, lv = rest_vals, current_vals, lock_vals
    n = len(rv)
    if n <= 1:
        return list(rv)

    mat = [[0.0, 0.0, 0.0] for _ in range(n)]
    res = [0.0] * n

    # Start case
    mat[0] = [0.0, -1.0, 1.0 - lv[0]]
    res[0] = -lv[0] * rv[0] + (1.0 - lv[0]) * (cv[1] - cv[0])

    # Mid cases
    for i in range(1, n - 1):
        A = (cv[i] - cv[i - 1]) / (cv[i + 1] - cv[i - 1])
        mat[i] = [(1.0 - lv[i]) * (1.0 - A), -1.0, (1.0 - lv[i]) * A]
        res[i] = -lv[i] * rv[i]

    # End case
    e = n - 1
    mat[e] = [1.0 - lv[e], -1.0, 0.0]
    res[e] = -lv[e] * rv[e] - (1.0 - lv[e]) * (cv[e] - cv[e - 1])

    return solve_tridiagonal(mat, res)


def solve_twist_param_matrix(rest_vals, current_vals, lock_vals):
    """Port of ``TwistSpline::solveTwistParamMatrix``.

    Same tridiagonal structure as ``solve_param_matrix`` but the right-hand
    side is purely the locked rest value (twists are handled differently from
    positional parameters).
    """
    rv, cv, lv = rest_vals, current_vals, lock_vals
    n = len(rv)
    if n <= 1:
        return list(rv)

    mat = [[0.0, 0.0, 0.0] for _ in range(n)]
    res = [0.0] * n

    mat[0] = [0.0, -1.0, 1.0 - lv[0]]
    res[0] = lv[0] * rv[0]

    for i in range(1, n - 1):
        A = (cv[i] - cv[i - 1]) / (cv[i + 1] - cv[i - 1])
        mat[i] = [(1.0 - lv[i]) * (1.0 - A), -1.0, (1.0 - lv[i]) * A]
        res[i] = lv[i] * rv[i]

    e = n - 1
    mat[e] = [1.0 - lv[e], -1.0, 0.0]
    res[e] = lv[e] * rv[e]

    return solve_tridiagonal(mat, res)
