"""
test_DenseDiscMoment.py

Tests for the moment methods of DenseDiscretePhaseType.

Discrete counterpart of jMarkov's ``test/jphase/DenseContMomentTest.java``:
same variable (the uniformization A = I + T/10 of Java's 11-phase matrix,
with Java's vector) and the same methods (expected value, variance, CV,
moments 2 to 4), each followed by the "Matrix changed" / "Vector changed"
checks.

The expected values come from two independent sources:

1. ``reference_values.MOMENT``: exact rational factorial moments
   k! alpha (I-A)^{-k} A^{k-1} 1, converted to raw moments with Stirling
   numbers and cross-checked against the series sum_k k^j P(X = k)
   (``generate_reference_values.py``).
2. ``reference_values.JAVA_CONT_MOMENTS``: realEV, realVar, realCV, realM2,
   realM3 and realM4 as hard-coded in DenseContMomentTest.java. The
   continuous variable is a sum of N independent Exp(lambda) times, with N
   this discrete variable, so T | N ~ Gamma(N, lambda) and

       E[T^k] = E[N (N+1) ... (N+k-1)] / lambda^k.

   The rising factorial moment on the right is built from this package's raw
   moments, so jMarkov's continuous values check the discrete moments
   directly, with jMarkov's tolerance (1.0E-5).

Naming: Java's ``CV()`` returns Var/E^2, the SQUARED coefficient of
variation. Here that is ``scv()``; ``cv()`` is the true sd/mean.

Run with:
    python3 -m unittest test_DenseDiscMoment -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import numpy as np

import discrete_fixtures as fx
from DenseDiscPhaseVar import DenseDiscretePhaseType
from matrix_utils import stirling_second, to_dense
from reference_values import JAVA_CONT_MOMENTS, MOMENT

JAVA_TOL = 1.0e-5


def rel_delta(x, rel=1e-10):
    return rel * max(1.0, abs(x))


class DenseDiscMomentTest(unittest.TestCase):
    """Mirror of DenseContMomentTest.java for the discrete case."""

    @staticmethod
    def make(vector, matrix):
        return DenseDiscretePhaseType(vector, matrix)

    def setUp(self):
        self.vector, self.matrix = fx.fixture("moment")
        self.var = self.make(self.vector, self.matrix)

    def tearDown(self):
        self.assertLess(np.max(np.abs(to_dense(self.var.A) - self.matrix)), 1.0e-12,
                        "Matrix changed")
        self.assertLess(np.max(np.abs(self.var.alpha - self.vector)), 1.0e-12,
                        "Vector changed")

    def test_expected_value(self):
        """Java: testExpectedValue."""
        self.assertAlmostEqual(self.var.mean(), MOMENT["mean"],
                               delta=rel_delta(MOMENT["mean"]),
                               msg="Expected Value non equal")

    def test_variance(self):
        """Java: testVariance."""
        self.assertAlmostEqual(self.var.var(), MOMENT["var"],
                               delta=rel_delta(MOMENT["var"]),
                               msg="Variance non equal")

    def test_std(self):
        self.assertAlmostEqual(self.var.std(), MOMENT["std"],
                               delta=rel_delta(MOMENT["std"]),
                               msg="Standard Deviation non equal")

    def test_cv(self):
        """Java: testCV (Java's CV is the SCV; both are checked)."""
        self.assertAlmostEqual(self.var.cv(), MOMENT["cv"],
                               delta=rel_delta(MOMENT["cv"]),
                               msg="Coefficient of Variation non equal")
        self.assertAlmostEqual(self.var.scv(), MOMENT["scv"],
                               delta=rel_delta(MOMENT["scv"]),
                               msg="Squared Coefficient of Variation non equal")

    def test_moment2(self):
        """Java: testMoment2."""
        self.assertAlmostEqual(self.var.moment(2), MOMENT["raw"][2],
                               delta=rel_delta(MOMENT["raw"][2]),
                               msg="Second Moment non equal")

    def test_moment3(self):
        """Java: testMoment3."""
        self.assertAlmostEqual(self.var.moment(3), MOMENT["raw"][3],
                               delta=rel_delta(MOMENT["raw"][3]),
                               msg="Third Moment non equal")

    def test_moment4(self):
        """Java: testMoment4."""
        self.assertAlmostEqual(self.var.moment(4), MOMENT["raw"][4],
                               delta=rel_delta(MOMENT["raw"][4]),
                               msg="Fourth Moment non equal")

    def test_factorial_moments(self):
        for k in range(1, 5):
            self.assertAlmostEqual(self.var.factorial_moment(k), MOMENT["factorial"][k],
                                   delta=rel_delta(MOMENT["factorial"][k]),
                                   msg=f"Factorial Moment {k} non equal")

    def test_java_continuous_moments(self):
        """
        jMarkov's DenseContMomentTest values, recovered from the discrete
        moments through E[T^k] = E[N(N+1)...(N+k-1)] / lambda^k.
        """
        lam = fx.MOMENT_LAMBDA
        raw = {k: self.var.moment(k) for k in range(1, 5)}
        # Rising factorial x^(k) = sum_j c(k, j) x^j, with c the unsigned
        # Stirling numbers of the first kind: x^(2) = x^2 + x, etc.
        unsigned_first = {1: {1: 1}, 2: {1: 1, 2: 1}, 3: {1: 2, 2: 3, 3: 1},
                          4: {1: 6, 2: 11, 3: 6, 4: 1}}
        cont = {k: sum(c * raw[j] for j, c in unsigned_first[k].items()) / lam ** k
                for k in range(1, 5)}
        cont_var = cont[2] - cont[1] ** 2
        self.assertAlmostEqual(cont[1], JAVA_CONT_MOMENTS["mean"], delta=JAVA_TOL,
                               msg="Expected Value non equal")
        self.assertAlmostEqual(cont_var, JAVA_CONT_MOMENTS["var"], delta=JAVA_TOL,
                               msg="Variance non equal")
        self.assertAlmostEqual(cont_var / cont[1] ** 2, JAVA_CONT_MOMENTS["cv"],
                               delta=JAVA_TOL, msg="Coefficient of Variance non equal")
        for k in (2, 3, 4):
            self.assertAlmostEqual(cont[k], JAVA_CONT_MOMENTS[f"m{k}"], delta=JAVA_TOL,
                                   msg=f"Moment {k} non equal")

    def test_raw_from_factorial_uses_stirling(self):
        fm = [self.var.factorial_moment(j) for j in range(1, 5)]
        expected = sum(stirling_second(4, j) * fm[j - 1] for j in range(1, 5))
        self.assertAlmostEqual(self.var.moment(4), expected, delta=rel_delta(expected))


if __name__ == "__main__":
    unittest.main()
