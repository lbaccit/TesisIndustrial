"""
test_DenseDiscProb.py

Tests for the probability methods of DenseDiscretePhaseType.

Discrete counterpart of jMarkov's ``test/jphase/DenseContProbTest.java``: same
three variables (var1, var2, var3), same structure (one test per variable,
followed by the "Matrix changed" / "Vector changed" checks), for the methods
of the discrete interface (pmf, cdf, survival, quantile).

The three variables are the uniformizations A = I + T/lambda of Java's
matrix1, matrix2 and matrix3 (see ``discrete_fixtures.py``), with Java's
vectors. The expected values come from two independent sources:

1. ``reference_values.PROB``: pmf, cdf and quantiles computed in exact
   rational arithmetic by stepping the absorbing chain
   (``generate_reference_values.py``).
2. ``reference_values.JAVA_CONT_PROB``: the realPDF/realCDF tables hard-coded
   in DenseContProbTest.java. If N is the discrete variable obtained by
   uniformizing T at rate lambda, the continuous variable is a sum of N
   independent Exp(lambda) times, so

       F_T(t) = sum_{n>=0} P(N = n) P(Poisson(lambda t) >= n)
       f_T(t) = sum_{n>=1} P(N = n) lambda P(Poisson(lambda t) = n-1)

   Evaluating the right-hand side with this package's pmf must reproduce
   jMarkov's tables, with jMarkov's own tolerance (1.0E-5).

Run with:
    python3 -m unittest test_DenseDiscProb -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import poisson

import discrete_fixtures as fx
from DenseDiscPhaseVar import DenseDiscretePhaseType
from matrix_utils import to_dense
from reference_values import JAVA_CONT_PROB, PROB

EXACT_TOL = 1e-12
JAVA_TOL = 1.0e-5
N_MAX = 3000


def continuous_from_discrete(var, lam, t):
    """pdf and cdf at times t of the sum of N ~ var independent Exp(lam)."""
    p = var.pmf_range(N_MAX)
    n = np.arange(N_MAX + 1)
    lt = lam * np.asarray(t, dtype=float)[:, None]
    cdf = (poisson.sf(n - 1, lt) * p).sum(axis=1)
    pdf = (lam * poisson.pmf(n[1:] - 1, lt) * p[1:]).sum(axis=1)
    return pdf, cdf


class DenseDiscProbTest(unittest.TestCase):
    """Mirror of DenseContProbTest.java for the discrete case."""

    @staticmethod
    def make(vector, matrix):
        return DenseDiscretePhaseType(vector, matrix)

    def setUp(self):
        self.vector1, self.matrix1 = fx.fixture("prob1")
        self.vector2, self.matrix2 = fx.fixture("prob2")
        self.vector3, self.matrix3 = fx.fixture("prob3")
        self.var1 = self.make(self.vector1, self.matrix1)
        self.var2 = self.make(self.vector2, self.matrix2)
        self.var3 = self.make(self.vector3, self.matrix3)
        self.lambdas = {"prob1": fx.PROB_LAMBDA1, "prob2": fx.PROB_LAMBDA2,
                        "prob3": fx.PROB_LAMBDA3}

    def assert_unchanged(self, var, vector, matrix):
        self.assertLess(np.max(np.abs(to_dense(var.A) - matrix)), 1.0e-12,
                        "Matrix changed")
        self.assertLess(np.max(np.abs(var.alpha - vector)), 1.0e-12,
                        "Vector changed")

    def check_prob(self, var, name):
        ref = PROB[name]
        for k, (real_pmf, real_cdf) in enumerate(zip(ref["pmf"], ref["cdf"])):
            self.assertAlmostEqual(var.pmf(k), real_pmf, delta=EXACT_TOL,
                                   msg=f"Probability Mass Function non equal (k={k})")
            self.assertAlmostEqual(var.cdf(k), real_cdf, delta=EXACT_TOL,
                                   msg=f"Cumulative Probability Function non equal (k={k})")
            self.assertAlmostEqual(var.survival(k), 1 - real_cdf, delta=EXACT_TOL,
                                   msg=f"Survival Function non equal (k={k})")
        k_max = len(ref["pmf"]) - 1
        assert_allclose(var.pmf_range(k_max), ref["pmf"], atol=EXACT_TOL,
                        err_msg="pmf_range non equal")
        assert_allclose(var.cdf_range(k_max), ref["cdf"], atol=EXACT_TOL,
                        err_msg="cdf_range non equal")

    def check_quantile(self, var, name):
        for p, k in PROB[name]["quantile"].items():
            self.assertEqual(var.quantile(p), k, f"Quantile non equal (p={p})")

    def check_java_tables(self, var, name):
        table = JAVA_CONT_PROB[name]
        t = np.array(table["index"]) * float(table["step"])
        pdf, cdf = continuous_from_discrete(var, self.lambdas[name], t)
        assert_allclose(pdf, table["pdf"], atol=JAVA_TOL, rtol=0,
                        err_msg="Probability Density Function non equal")
        assert_allclose(cdf, table["cdf"], atol=JAVA_TOL, rtol=0,
                        err_msg="Cumulative Probability Function non equal")

    def test_prob1(self):
        """Java: testProb1."""
        self.check_prob(self.var1, "prob1")
        self.assert_unchanged(self.var1, self.vector1, self.matrix1)

    def test_prob2(self):
        """Java: testProb2."""
        self.check_prob(self.var2, "prob2")
        self.assert_unchanged(self.var2, self.vector2, self.matrix2)

    def test_prob3(self):
        """Java: testProb3."""
        self.check_prob(self.var3, "prob3")
        self.assert_unchanged(self.var3, self.vector3, self.matrix3)

    def test_quantile1(self):
        self.check_quantile(self.var1, "prob1")
        self.assertEqual(self.var1.median(), PROB["prob1"]["quantile"][0.5])
        self.assert_unchanged(self.var1, self.vector1, self.matrix1)

    def test_quantile2(self):
        self.check_quantile(self.var2, "prob2")
        self.assert_unchanged(self.var2, self.vector2, self.matrix2)

    def test_quantile3(self):
        self.check_quantile(self.var3, "prob3")
        self.assert_unchanged(self.var3, self.vector3, self.matrix3)

    def test_java_tables1(self):
        """Reproduces the 151 realPDF/realCDF values of Java's testProb1."""
        self.check_java_tables(self.var1, "prob1")
        self.assert_unchanged(self.var1, self.vector1, self.matrix1)

    def test_java_tables2(self):
        """Reproduces the 151 realPDF/realCDF values of Java's testProb2."""
        self.check_java_tables(self.var2, "prob2")
        self.assert_unchanged(self.var2, self.vector2, self.matrix2)

    def test_java_tables3(self):
        """Reproduces Java's testProb3 table (every 40th of its 4001 values)."""
        self.check_java_tables(self.var3, "prob3")
        self.assert_unchanged(self.var3, self.vector3, self.matrix3)

    def test_pmf_sums_to_one(self):
        for var in (self.var1, self.var2, self.var3):
            self.assertAlmostEqual(var.pmf_range(N_MAX).sum(), 1.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
