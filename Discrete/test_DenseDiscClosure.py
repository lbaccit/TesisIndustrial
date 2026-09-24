"""
test_DenseDiscClosure.py

Tests for the closure methods of DenseDiscretePhaseType: sum, min, max, mix,
sum_geom and sum_ph.

Discrete counterpart of jMarkov's ``test/jphase/DenseContClosureTest.java``:
same variables (var1, var2 as uniformizations of Java's matrix1/matrix2, with
Java's vectors, and varD, which is Java's own discrete variable
matrixD/vectorD), same operations and parameters (mix(0.2, ...),
sumGeom(0.2), var2.sumPH(varD)). Java's residualTime, eqResidualTime and
waitingQ are continuous-only and have no discrete counterpart.

Three layers of checks:

1. ``DenseDiscClosureTest`` - like Java: the (alpha, A) produced by each
   operation must equal the hand-built representation
   (``reference_values.CLOSURE``, exact arithmetic) and the operands must not
   change.
2. ``DenseDiscClosureDistributionTest`` - the resulting DISTRIBUTION must
   satisfy identities that hold for any independent random variables,
   phase-type or not: convolution for the sum, 1-(1-F)(1-G) and F*G for min
   and max, the p-weighted pmf for the mixture, and brute-force compound sums
   for sum_geom and sum_ph. Several fixtures have alpha_0 > 0 (mass at 0),
   which is where the geometric and PH-counted sums are delicate.
3. ``DenseDiscClosureJavaDivergenceTest`` - the four places where the Java
   formula is wrong for the discrete case (documented in the docstrings of
   ``max``, ``sum_geom`` and ``sum_ph``): each test rebuilds Java's formula,
   shows that it fails, and shows that this implementation does not.

Run with:
    python3 -m unittest test_DenseDiscClosure -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

import discrete_fixtures as fx
from DenseDiscPhaseVar import DenseDiscretePhaseType
from matrix_utils import kronecker_sum, to_dense
from reference_values import CLOSURE

TOL = 1e-12
K = 60


def compound_pmf(pmf_x, pmf_n, k_max=K):
    """pmf of X_1 + ... + X_N (X_i iid with pmf_x, N with pmf_n), by brute force."""
    out = np.zeros(k_max + 1)
    conv_n = np.zeros(k_max + 1)
    conv_n[0] = 1.0                       # X^{*0}: point mass at 0
    for w in pmf_n:
        out += w * conv_n
        conv_n = np.convolve(conv_n, pmf_x)[:k_max + 1]
    return out


class ClosureFixtures(unittest.TestCase):
    """setUp shared by the three test classes below."""

    @staticmethod
    def make(vector, matrix):
        return DenseDiscretePhaseType(vector, matrix)

    def setUp(self):
        self.vector1, self.matrix1 = fx.fixture("closure1")
        self.vector2, self.matrix2 = fx.fixture("closure2")
        self.vectorD, self.matrixD = fx.fixture("closureD")
        self.var1 = self.make(self.vector1, self.matrix1)
        self.var2 = self.make(self.vector2, self.matrix2)
        self.varD = self.make(self.vectorD, self.matrixD)

    def tearDown(self):
        for var, vector, matrix in ((self.var1, self.vector1, self.matrix1),
                                    (self.var2, self.vector2, self.matrix2),
                                    (self.varD, self.vectorD, self.matrixD)):
            self.assertLess(np.max(np.abs(to_dense(var.A) - matrix)), 1.0e-12,
                            "Matrix changed")
            self.assertLess(np.max(np.abs(var.alpha - vector)), 1.0e-12,
                            "Vector changed")


class DenseDiscClosureTest(ClosureFixtures):
    """Mirror of DenseContClosureTest.java for the discrete case."""

    def check(self, calc, name, label):
        real = CLOSURE[name]
        assert_allclose(to_dense(calc.A), real["A"], atol=TOL, rtol=0,
                        err_msg=f"{label} non equal (Matrix)")
        assert_allclose(calc.alpha, real["alpha"], atol=TOL, rtol=0,
                        err_msg=f"{label} non equal (Vector)")
        self.assertIsInstance(calc, type(self.var1))

    def test_sum(self):
        """Java: testSum."""
        self.check(self.var1.sum(self.var2), "sum", "Sum of Variables")

    def test_min(self):
        """Java: testMin."""
        self.check(self.var1.min(self.var2), "min", "Min of Variables")

    def test_max(self):
        """Java: testMax."""
        self.check(self.var1.max(self.var2), "max", "Max of Variables")

    def test_mix(self):
        """Java: testMix."""
        self.check(self.var1.mix(fx.MIX_P, self.var2), "mix", "Mix of Variables")

    def test_sum_geom(self):
        """Java: testSumGeom."""
        self.check(self.var2.sum_geom(fx.SUM_GEOM_P), "sum_geom",
                   "Geometric Sum of Variables")

    def test_sum_ph(self):
        """Java: testSumPH."""
        self.check(self.var2.sum_ph(self.varD), "sum_ph",
                   "Sum of a discrete phase number")


class DenseDiscClosureDistributionTest(ClosureFixtures):
    """The resulting distributions satisfy the probability identities."""

    def test_sum_is_convolution(self):
        for X, Y in ((self.var1, self.var2), (self.var2, self.varD),
                     (self.varD, self.var1)):
            conv = np.convolve(X.pmf_range(K), Y.pmf_range(K))[:K + 1]
            assert_allclose(X.sum(Y).pmf_range(K), conv, atol=TOL)
            self.assertAlmostEqual(X.sum(Y).mean(), X.mean() + Y.mean(), delta=1e-10)

    def test_min_and_max_cdf_identities(self):
        for X, Y in ((self.var1, self.var2), (self.var2, self.varD)):
            F, G = X.cdf_range(K), Y.cdf_range(K)
            assert_allclose(X.min(Y).cdf_range(K), 1 - (1 - F) * (1 - G), atol=TOL)
            assert_allclose(X.max(Y).cdf_range(K), F * G, atol=TOL)

    def test_mix_is_weighted_pmf(self):
        p = fx.MIX_P
        mixed = self.var1.mix(p, self.var2)
        assert_allclose(mixed.pmf_range(K),
                        p * self.var1.pmf_range(K) + (1 - p) * self.var2.pmf_range(K),
                        atol=TOL)

    def test_sum_geom_is_compound_geometric_sum(self):
        p = fx.SUM_GEOM_P
        pmf_n = [0.0] + [p * (1 - p) ** (n - 1) for n in range(1, 400)]
        for X in (self.var1, self.var2, self.varD):
            assert_allclose(X.sum_geom(p).pmf_range(K),
                            compound_pmf(X.pmf_range(K), pmf_n), atol=1e-10)
            # Wald's identity: E[X_1 + ... + X_N] = E[N] E[X] = E[X] / p.
            self.assertAlmostEqual(X.sum_geom(p).mean(), X.mean() / p, delta=1e-9)

    def test_sum_ph_is_compound_sum(self):
        pmf_n = self.varD.pmf_range(400)
        for X in (self.var1, self.var2):
            assert_allclose(X.sum_ph(self.varD).pmf_range(K),
                            compound_pmf(X.pmf_range(K), pmf_n), atol=1e-10)
            self.assertAlmostEqual(X.sum_ph(self.varD).mean(),
                                   X.mean() * self.varD.mean(), delta=1e-9)

    def test_sum_ph_with_geometric_counter_is_sum_geom(self):
        p = fx.SUM_GEOM_P
        counter = self.make([1.0], [[1 - p]])
        assert_allclose(self.var2.sum_ph(counter).pmf_range(K),
                        self.var2.sum_geom(p).pmf_range(K), atol=TOL)

    def test_sum_ph_with_counter_one_is_identity(self):
        one = self.make([1.0], [[0.0]])
        assert_allclose(self.var2.sum_ph(one).pmf_range(K),
                        self.var2.pmf_range(K), atol=TOL)

    def test_phase_counts(self):
        n1, n2, nD = self.var1.n_phases, self.var2.n_phases, self.varD.n_phases
        self.assertEqual(self.var1.sum(self.var2).n_phases, n1 + n2)
        self.assertEqual(self.var1.mix(0.5, self.var2).n_phases, n1 + n2)
        self.assertEqual(self.var1.min(self.var2).n_phases, n1 * n2)
        self.assertEqual(self.var1.max(self.var2).n_phases, n1 * n2 + n1 + n2)
        self.assertEqual(self.var2.sum_geom(0.5).n_phases, n2)
        self.assertEqual(self.var2.sum_ph(self.varD).n_phases, n2 * nD)

    def test_sum_with_identically_zero_variable(self):
        zero = self.make([0.0, 0.0], self.matrix1)
        self.assertIs(zero.sum(self.var2), self.var2)
        self.assertIs(self.var2.sum(zero), self.var2)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            self.var1.mix(1.5, self.var2)
        with self.assertRaises(ValueError):
            self.var1.sum_geom(0.0)


class DenseDiscClosureJavaDivergenceTest(ClosureFixtures):
    """Java's discrete formulas, rebuilt here, fail where this code does not."""

    def test_max_java_formula_is_not_substochastic(self):
        """
        Java's discrete max() copies the continuous one: Kronecker SUM for
        the joint block and identities at the boundary. In discrete time that
        gives rows summing to more than 1.
        """
        X, Y = self.varD, self.varD
        n1, n2 = X.n_phases, Y.n_phases
        java_joint = kronecker_sum(to_dense(X.A), to_dense(Y.A))
        java_edges = np.hstack([np.kron(np.eye(n1), Y.get_mat0()[:, None]),
                                np.kron(X.get_mat0()[:, None], np.eye(n2))])
        java_rows = np.hstack([java_joint, java_edges]).sum(axis=1)
        self.assertGreater(java_rows.max(), 1.0 + 1e-6)
        ours = to_dense(X.max(Y).A)
        self.assertTrue(np.all(ours.sum(axis=1) <= 1.0 + 1e-12))

    def test_sum_ph_works_when_phase_counts_differ(self):
        """Java inverts I - a0 S against I_n1: it only runs when n1 == n2."""
        self.assertNotEqual(self.var2.n_phases, self.varD.n_phases)
        self.assertEqual(self.var2.sum_ph(self.varD).n_phases, 3 * 2)

    def test_sum_geom_java_formula_fails_with_mass_at_zero(self):
        """Java: alpha_res = alpha, A_res = A + (1-p) a alpha (no 1/(1-(1-p)a0))."""
        p, X = fx.SUM_GEOM_P, self.var2
        self.assertGreater(X.get_vec0(), 0)
        java = self.make(X.alpha, to_dense(X.A) + (1 - p) * np.outer(X.get_mat0(), X.alpha))
        pmf_n = [0.0] + [p * (1 - p) ** (n - 1) for n in range(1, 400)]
        truth = compound_pmf(X.pmf_range(K), pmf_n)
        self.assertGreater(np.max(np.abs(java.pmf_range(K) - truth)), 1e-3)
        assert_allclose(X.sum_geom(p).pmf_range(K), truth, atol=1e-10)

    def test_sum_ph_java_formula_fails_with_mass_at_zero(self):
        """Java multiplies the restart term by an extra (1 - a0)."""
        X, N = self.var2, self.varD
        a0 = X.get_vec0()
        self.assertGreater(a0, 0)
        S = to_dense(N.A)
        M = np.linalg.inv(np.eye(N.n_phases) - a0 * S)
        java = self.make(np.kron(X.alpha, M.T @ N.alpha),
                         np.kron(to_dense(X.A), np.eye(N.n_phases))
                         + np.kron((1 - a0) * np.outer(X.get_mat0(), X.alpha), M @ S))
        truth = compound_pmf(X.pmf_range(K), N.pmf_range(400))
        self.assertGreater(np.max(np.abs(java.pmf_range(K) - truth)), 1e-3)
        assert_allclose(X.sum_ph(N).pmf_range(K), truth, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
