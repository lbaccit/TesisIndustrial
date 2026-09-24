"""
discrete_fixtures.py

Discrete Phase-Type variables used by the test suite, mirroring the fixtures
of jMarkov's own jphase tests.

jMarkov only has tests for the CONTINUOUS case (``DenseContProbTest``,
``DenseContMomentTest``, ``DenseContClosureTest`` and their Sparse twins in
``jMarkov/test/jphase/``). To keep the discrete suite structurally identical,
each continuous fixture T (a sub-generator) is turned into a discrete one by
uniformization,

    A = I + T / lambda,        lambda = max_i |T_ii|,

which is the standard link between continuous- and discrete-time chains
(Latouche & Ramaswami, 1999). A is non-negative, its rows sum to 1 + (T 1)_i /
lambda <= 1, and it leaks exactly from the phases where T leaks, so it is a
valid transient sub-stochastic matrix with the same structure as T. The
initial vectors are Java's, unchanged (several have alpha_0 > 0, a mass at 0,
which is exactly the case the closure formulas have to handle).

``CLOSURE_VECTOR_D``/``CLOSURE_MATRIX_D`` need no conversion: it is the
discrete variable that ``DenseContClosureTest`` itself builds
(``DenseDiscPhaseVar(vectorD, matrixD)``) for ``testSumPH``.

Every value is a decimal literal with an exact rational meaning, so
``generate_reference_values.py`` can rebuild the same matrices in exact
arithmetic with ``fractions.Fraction``.

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

from fractions import Fraction

import numpy as np


def uniformize(T, lam, exact: bool = False):
    """A = I + T / lam, as floats or as exact Fractions."""
    n = len(T)
    if exact:
        lam = Fraction(str(lam))
        return [[(1 if i == j else 0) + Fraction(str(T[i][j])) / lam
                 for j in range(n)] for i in range(n)]
    return np.eye(n) + np.asarray(T, dtype=float) / float(lam)


# -----------------------------------------------------------------------------
# DenseContProbTest.java: matrix1/vector1, matrix2/vector2, matrix3/vector3
# -----------------------------------------------------------------------------

PROB_T1 = [[-2, 2],
           [2, -5]]
PROB_LAMBDA1 = 5
PROB_VECTOR1 = [0.2, 0.4]

PROB_T2 = [
    [-6, 2, 1, 2, 0, 0, 1, 0, 0, 0, 0],
    [1, -5, 1, 0, 2, 0, 1, 0, 0, 0, 0],
    [2, 1, -7, 0, 0, 2, 2, 0, 0, 0, 0],
    [2, 0, 0, -9, 2, 1, 0, 1, 3, 0, 0],
    [0, 2, 0, 1, -8, 1, 0, 1, 0, 3, 0],
    [0, 0, 2, 2, 1, -10, 2, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, -2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, -5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -4, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, -3, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, -5],
]
PROB_LAMBDA2 = 10
PROB_VECTOR2 = [0.2, 0.3, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 0]

PROB_T3 = [[-0.1, 0.09, 0, 0],
           [0, -0.01, 0.01, 0],
           [0, 0, -0.01, 0.01],
           [0, 0, 0, -0.01]]
PROB_LAMBDA3 = 0.1
PROB_VECTOR3 = [1, 0, 0, 0]

# -----------------------------------------------------------------------------
# DenseContMomentTest.java: matrix/vector
# -----------------------------------------------------------------------------

MOMENT_T = [
    [-6, 2, 1, 2, 0, 0, 1, 0, 0, 0, 0],
    [1, -5, 1, 0, 2, 0, 1, 0, 0, 0, 0],
    [2, 1, -7, 0, 0, 2, 2, 0, 0, 0, 0],
    [2, 0, 0, -9, 2, 1, 0, 1, 3, 0, 0],
    [0, 2, 0, 1, -8, 1, 0, 1, 0, 3, 0],
    [0, 0, 2, 2, 1, -10, 0, 2, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, -2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, -5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -4, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, -3, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, -5],
]
MOMENT_LAMBDA = 10
MOMENT_VECTOR = [0.02, 0.04, 0.04, 0.04, 0.08, 0.08, 0.1, 0.2, 0.04, 0.08, 0.08]

# -----------------------------------------------------------------------------
# DenseContClosureTest.java: matrix1/vector1, matrix2/vector2, matrixD/vectorD
# -----------------------------------------------------------------------------

CLOSURE_T1 = [[-2, 2],
              [2, -5]]
CLOSURE_LAMBDA1 = 5
CLOSURE_VECTOR1 = [0.2, 0.4]

CLOSURE_T2 = [[-4, 2, 1],
              [1, -3, 1],
              [2, 1, -5]]
CLOSURE_LAMBDA2 = 5
CLOSURE_VECTOR2 = [0.1, 0.2, 0.2]

CLOSURE_MATRIX_D = [[0.76, 0.24],
                    [0, 0.76]]
CLOSURE_VECTOR_D = [0.15, 0.85]

# Parameters used by DenseContClosureTest: mix(0.2, ...) and sumGeom(0.2).
MIX_P = 0.2
SUM_GEOM_P = 0.2


def dph(vector, T=None, lam=None, matrix=None, exact: bool = False):
    """(alpha, A) for a fixture, either uniformized from T or given directly."""
    if exact:
        alpha = [Fraction(str(x)) for x in vector]
        A = (uniformize(T, lam, exact=True) if matrix is None
             else [[Fraction(str(x)) for x in row] for row in matrix])
        return alpha, A
    alpha = np.asarray(vector, dtype=float)
    A = uniformize(T, lam) if matrix is None else np.asarray(matrix, dtype=float)
    return alpha, A


FIXTURES = {
    "prob1": dict(vector=PROB_VECTOR1, T=PROB_T1, lam=PROB_LAMBDA1),
    "prob2": dict(vector=PROB_VECTOR2, T=PROB_T2, lam=PROB_LAMBDA2),
    "prob3": dict(vector=PROB_VECTOR3, T=PROB_T3, lam=PROB_LAMBDA3),
    "moment": dict(vector=MOMENT_VECTOR, T=MOMENT_T, lam=MOMENT_LAMBDA),
    "closure1": dict(vector=CLOSURE_VECTOR1, T=CLOSURE_T1, lam=CLOSURE_LAMBDA1),
    "closure2": dict(vector=CLOSURE_VECTOR2, T=CLOSURE_T2, lam=CLOSURE_LAMBDA2),
    "closureD": dict(vector=CLOSURE_VECTOR_D, matrix=CLOSURE_MATRIX_D),
}


def fixture(name: str, exact: bool = False):
    """(alpha, A) of the named fixture, as NumPy floats or exact Fractions."""
    return dph(exact=exact, **FIXTURES[name])
