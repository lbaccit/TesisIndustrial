"""
test_AbstractDistPhaseVar.py

Tests for AbstractDistPhaseVar.py, using DenseDiscretePhaseType as the concrete
implementation (the abstract class cannot be instantiated on its own).

About the reference values
--------------------------
jMarkov (Java) has NO tests for the discrete Phase-Type variable:
DenseContProbTest.java and DenseContMomentTest.java exist for the continuous
case, but there is no DenseDiscProbTest.java or DenseDiscMomentTest.java in the
repository (`jMarkov/test/jphase/`). So for the discrete case there are no
official Java "gold" values to copy here.

Instead, the verification rests on:

  1. Cases with a known closed-form solution in the Phase-Type literature
     (Neuts 1981; Latouche & Ramaswami 1999):
       - Geometric(p)          as a 1-phase DPH.
       - Negative Binomial(r,p) as r geometric phases in series.
  2. Brute force: summing pmf(k) over many k and comparing against 1, or
     building the cdf by accumulating the pmf term by term and comparing it
     against cdf() directly.
  3. The quantile values where Java's Newton-Raphson silently fails, reproduced
     exactly.

Run with:
    python3 -m unittest test_AbstractDistPhaseVar -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest
from math import comb

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from DenseDiscPhaseVar import DenseDiscretePhaseType as D
from SparseDiscPhaseVar import SparseDiscretePhaseType as S


# =============================================================================
# Fixtures: distributions with a known closed form
# =============================================================================

def geometric(p: float) -> D:
    """
    Geometric on {1, 2, 3, ...} with success parameter p.

    DPH(alpha=[1], A=[[1-p]]). E[X] = 1/p, Var(X) = (1-p)/p^2.
    """
    return D(np.array([1.0]), np.array([[1.0 - p]]))


def negative_binomial(r: int, p: float) -> D:
    """
    Negative Binomial(r, p): r geometric(p) phases in series (discrete Erlang).

    E[X] = r/p, Var(X) = r(1-p)/p^2.
    """
    q = 1.0 - p
    A = np.diag([q] * r) + np.diag([p] * (r - 1), k=1)
    alpha = np.zeros(r)
    alpha[0] = 1.0
    return D(alpha, A)


class TestGeometricAnalytic(unittest.TestCase):
    """Geometric(p=0.5): the simplest case, with a closed form for everything."""

    def setUp(self):
        self.p = 0.5
        self.X = geometric(self.p)

    def test_mean(self):
        self.assertAlmostEqual(self.X.mean(), 1 / self.p, places=10)

    def test_var(self):
        expected = (1 - self.p) / self.p ** 2
        self.assertAlmostEqual(self.X.var(), expected, places=10)

    def test_pmf_matches_closed_form(self):
        for k in range(1, 10):
            expected = (1 - self.p) ** (k - 1) * self.p
            self.assertAlmostEqual(self.X.pmf(k), expected, places=10)

    def test_pmf_at_zero_is_zero(self):
        # Classic geometric: support on {1, 2, 3, ...}, so P(X=0) = 0.
        self.assertAlmostEqual(self.X.pmf(0), 0.0, places=10)

    def test_cdf_matches_closed_form(self):
        for k in range(1, 10):
            expected = 1 - (1 - self.p) ** k
            self.assertAlmostEqual(self.X.cdf(k), expected, places=10)

    def test_survival_is_1_minus_cdf(self):
        for k in range(1, 10):
            self.assertAlmostEqual(self.X.survival(k), 1 - self.X.cdf(k),
                                   places=10)

    def test_pmf_sums_to_one(self):
        total = sum(self.X.pmf(k) for k in range(200))
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_quantile_matches_direct_search(self):
        """
        Independent reference for the quantile: search directly for the
        smallest k with cdf(k) >= p, without going through the internal
        accumulation method.
        """
        for p_order in (0.5, 0.75, 0.9, 0.95, 0.99):
            computed = self.X.quantile(p_order)
            reference = next(k for k in range(1, 1000)
                             if self.X.cdf(k) >= p_order - 1e-12)
            self.assertEqual(computed, reference,
                             f"quantile({p_order}) disagrees with direct search")

    def test_median_equals_quantile_one_half(self):
        self.assertEqual(self.X.median(), self.X.quantile(0.5))

    def test_quantile_does_not_reproduce_java_bug(self):
        """
        Documented in AbstractDistPhaseVar.py: Java's quantil() applies
        Newton-Raphson and silently returns 0.0 when it fails to converge
        within 100 iterations. For Geometric(0.5) that happens at p=0.9, 0.95
        and 0.99, where the correct values are 4, 5 and 7 respectively.
        """
        self.assertEqual(self.X.quantile(0.9), 4)
        self.assertEqual(self.X.quantile(0.95), 5)
        self.assertEqual(self.X.quantile(0.99), 7)


class TestNegativeBinomialAnalytic(unittest.TestCase):
    """Negative Binomial(r=2, p=0.4): two phases in series."""

    def setUp(self):
        self.r, self.p = 2, 0.4
        self.X = negative_binomial(self.r, self.p)

    def test_mean(self):
        self.assertAlmostEqual(self.X.mean(), self.r / self.p, places=8)

    def test_var(self):
        expected = self.r * (1 - self.p) / self.p ** 2
        self.assertAlmostEqual(self.X.var(), expected, places=8)

    def test_pmf_matches_negative_binomial_formula(self):
        q = 1 - self.p
        for k in range(self.r, self.r + 10):
            expected = comb(k - 1, self.r - 1) * (self.p ** self.r) * (q ** (k - self.r))
            self.assertAlmostEqual(self.X.pmf(k), expected, places=8)

    def test_pmf_zero_below_support(self):
        # The support starts at k=r (at least two phases to reach absorption).
        self.assertAlmostEqual(self.X.pmf(0), 0.0, places=10)
        self.assertAlmostEqual(self.X.pmf(1), 0.0, places=10)

    def test_pmf_sums_to_one(self):
        total = sum(self.X.pmf(k) for k in range(500))
        self.assertAlmostEqual(total, 1.0, places=6)


class TestPmfCdfConsistency(unittest.TestCase):
    """The hand-accumulated cdf must match cdf() for any DPH."""

    def setUp(self):
        alpha = np.array([0.4, 0.3, 0.3])
        A = np.array([[0.5, 0.2, 0.0],
                      [0.1, 0.6, 0.1],
                      [0.0, 0.0, 0.4]])
        self.X = D(alpha, A)

    def test_cdf_equals_cumulative_pmf(self):
        running = 0.0
        for k in range(50):
            running += self.X.pmf(k)
            self.assertAlmostEqual(self.X.cdf(k), running, places=8)

    def test_pmf_range_matches_individual_calls(self):
        block = self.X.pmf_range(20)
        one_by_one = np.array([self.X.pmf(k) for k in range(21)])
        assert_allclose(block, one_by_one, atol=1e-10)

    def test_cdf_range_matches_individual_calls(self):
        block = self.X.cdf_range(20)
        one_by_one = np.array([self.X.cdf(k) for k in range(21)])
        assert_allclose(block, one_by_one, atol=1e-10)

    def test_prob_between_a_and_b(self):
        # Convention of this implementation (see the prob() docstring):
        # P(a < X <= b) = F(b) - F(a), not P(a <= X <= b).
        a, b = 2, 5
        expected = self.X.cdf(b) - self.X.cdf(a)
        self.assertAlmostEqual(self.X.prob(a, b), expected, places=10)

    def test_prob_returns_zero_if_b_not_greater_than_a(self):
        self.assertEqual(self.X.prob(5, 5), 0.0)
        self.assertEqual(self.X.prob(5, 2), 0.0)

    def test_moment_1_equals_mean(self):
        self.assertAlmostEqual(self.X.moment(1), self.X.mean(), places=8)

    def test_std_squared_equals_var(self):
        self.assertAlmostEqual(self.X.std() ** 2, self.X.var(), places=8)

    def test_scv_equals_var_over_mean_squared(self):
        expected = self.X.var() / self.X.mean() ** 2
        self.assertAlmostEqual(self.X.scv(), expected, places=10)

    def test_cv_is_sqrt_scv(self):
        self.assertAlmostEqual(self.X.cv(), self.X.scv() ** 0.5, places=10)


class TestVec0Mat0Consistency(unittest.TestCase):
    """get_vec0/get_mat0 must be consistent with pmf(0) and with the cdf."""

    def setUp(self):
        alpha = np.array([0.4, 0.3])
        A = np.array([[0.5, 0.2], [0.1, 0.6]])
        self.X = D(alpha, A)

    def test_get_vec0_equals_pmf_zero(self):
        self.assertAlmostEqual(self.X.get_vec0(), self.X.pmf(0), places=10)

    def test_get_mat0_is_absorption_vector(self):
        A = self.X.A
        expected = np.ones(2) - A @ np.ones(2)
        assert_allclose(self.X.get_mat0(), expected, atol=1e-10)


class TestConstructorValidation(unittest.TestCase):
    def test_rejects_non_square_matrix(self):
        with self.assertRaises(ValueError):
            D(np.array([1.0, 0.0]), np.zeros((2, 3)))

    def test_rejects_recurrent_matrix(self):
        # A row summing to exactly 1: absorption is impossible.
        with self.assertRaises(ValueError):
            D(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.5, 0.4]]))

    def test_rejects_negative_alpha(self):
        with self.assertRaises(ValueError):
            D(np.array([-0.1, 1.1]), np.array([[0.5, 0.0], [0.0, 0.5]]))

    def test_accepts_defective_alpha(self):
        # alpha_0 > 0 (mass on immediate absorption) is valid: P(X=0) > 0.
        X = D(np.array([0.5, 0.3]), np.array([[0.5, 0.2], [0.1, 0.6]]))
        self.assertGreater(X.get_vec0(), 0.0)


class TestEqualityAndHash(unittest.TestCase):
    def test_copy_is_equal_but_independent(self):
        X = D(np.array([0.4, 0.3]), np.array([[0.5, 0.2], [0.1, 0.6]]))
        Y = X.copy()
        self.assertEqual(X, Y)
        Y._alpha[0] = 0.0
        self.assertNotEqual(X._alpha[0], Y._alpha[0])

    def test_equal_objects_have_equal_hash(self):
        X = D(np.array([0.4, 0.3]), np.array([[0.5, 0.2], [0.1, 0.6]]))
        Y = X.copy()
        self.assertEqual(hash(X), hash(Y))

    def test_dense_and_sparse_equal_if_same_values(self):
        alpha = np.array([0.4, 0.3])
        A = np.array([[0.5, 0.2], [0.1, 0.6]])
        X_dense = D(alpha, A)
        X_sparse = D(alpha, sp.csr_array(A))
        self.assertEqual(X_dense, X_sparse)


class TestDenseVsSparseEndToEnd(unittest.TestCase):
    """The whole class, end to end, with a sparse matrix."""

    def setUp(self):
        alpha = np.array([0.4, 0.3])
        A = np.array([[0.5, 0.2], [0.1, 0.6]])
        self.dense = D(alpha, A)

    def test_all_public_methods_match(self):
        for api in (sp.csr_matrix, sp.csr_array):
            sparse_var = D(self.dense.alpha.copy(), api(self.dense.A))
            for method, arg in [
                ("mean", None), ("var", None), ("std", None), ("cv", None),
                ("scv", None), ("get_vec0", None),
                ("moment", 3), ("factorial_moment", 2),
                ("pmf", 4), ("cdf", 4), ("survival", 4),
            ]:
                a = getattr(self.dense, method)() if arg is None \
                    else getattr(self.dense, method)(arg)
                b = getattr(sparse_var, method)() if arg is None \
                    else getattr(sparse_var, method)(arg)
                self.assertAlmostEqual(float(np.asarray(a)), float(np.asarray(b)),
                                       places=10,
                                       msg=f"{method} differs with {api.__name__}")

    def test_get_mat0_matches(self):
        for api in (sp.csr_matrix, sp.csr_array):
            sparse_var = D(self.dense.alpha.copy(), api(self.dense.A))
            assert_allclose(self.dense.get_mat0(), sparse_var.get_mat0(), atol=1e-10)


class TestMinMax(unittest.TestCase):
    """
    min/max validated against direct probability identities, independent of
    the PH machinery:

        P(min(X,Y) <= k) = 1 - (1-F_X(k))(1-F_Y(k))
        P(max(X,Y) <= k) = F_X(k) * F_Y(k)

    These hold for ANY two independent random variables, so they do not rely
    on X, Y being phase-type at all - a strong, independent check.
    """

    def setUp(self):
        self.X = geometric(0.5)
        self.Y = geometric(0.7)
        alphaZ = np.array([1.0, 0.0])
        AZ = np.array([[0.6, 0.4], [0.0, 0.6]])
        self.Z = D(alphaZ, AZ)  # negative_binomial(2, 0.4), multi-phase

    def test_min_matches_survival_formula(self):
        mn = self.X.min(self.Y)
        for k in range(15):
            expected = 1 - (1 - self.X.cdf(k)) * (1 - self.Y.cdf(k))
            self.assertAlmostEqual(mn.cdf(k), expected, places=8)

    def test_max_matches_product_formula(self):
        mx = self.X.max(self.Y)
        for k in range(15):
            expected = self.X.cdf(k) * self.Y.cdf(k)
            self.assertAlmostEqual(mx.cdf(k), expected, places=8)

    def test_min_max_multiphase(self):
        mn = self.X.min(self.Z)
        mx = self.X.max(self.Z)
        for k in range(20):
            fx, fz = self.X.cdf(k), self.Z.cdf(k)
            self.assertAlmostEqual(mn.cdf(k), 1 - (1 - fx) * (1 - fz), places=8)
            self.assertAlmostEqual(mx.cdf(k), fx * fz, places=8)

    def test_phase_counts(self):
        n1, n2 = self.X.n_phases, self.Z.n_phases
        self.assertEqual(self.X.min(self.Z).n_phases, n1 * n2)
        self.assertEqual(self.X.max(self.Z).n_phases, n1 * n2 + n1 + n2)

    def test_max_row_sums_are_valid_after_java_bug_fix(self):
        """
        Regression test for the bug documented in ``max``'s docstring: Java's
        discrete ``max`` reuses the continuous-time formula (Kronecker SUM
        for the joint block, identity next to each ``mat0`` at the
        boundary), which produces invalid rows (sum > 1) whenever both
        variables have a real chance of continuing. This checks the fixed
        construction directly, without going through cdf().
        """
        mx = self.X.max(self.Y)
        A = mx.A
        row_sums = A.sum(axis=1)
        self.assertTrue(np.all(row_sums <= 1.0 + 1e-9))
        # Joint state (0,0): missing mass must equal mat0(X)[0]*mat0(Y)[0].
        m1, m2 = self.X.get_mat0(), self.Y.get_mat0()
        self.assertAlmostEqual(1 - row_sums[0], m1[0] * m2[0], places=10)


class TestSum(unittest.TestCase):
    """
    sum() validated against discrete convolution: if Z = X + Y with X, Y
    independent, pmf_Z = pmf_X * pmf_Y (convolution), regardless of whether
    the summands are phase-type.
    """

    def setUp(self):
        self.X = geometric(0.5)
        self.Y = negative_binomial(2, 0.4)

    def test_sum_matches_convolution(self):
        s = self.X.sum(self.Y)
        pmf_x = self.X.pmf_range(30)
        pmf_y = self.Y.pmf_range(30)
        conv = np.convolve(pmf_x, pmf_y)[:31]
        assert_allclose(s.pmf_range(30), conv, atol=1e-8)

    def test_sum_mean_is_additive(self):
        s = self.X.sum(self.Y)
        self.assertAlmostEqual(s.mean(), self.X.mean() + self.Y.mean(), places=8)

    def test_sum_phase_count(self):
        s = self.X.sum(self.Y)
        self.assertEqual(s.n_phases, self.X.n_phases + self.Y.n_phases)

    def test_degenerate_zero_variable_is_identity(self):
        # alpha entirely zero => P(X=0)=1 identically; Java returns the other
        # operand unchanged rather than computing anything.
        zero_var = D(np.array([0.0]), np.array([[0.5]]))
        self.assertIs(zero_var.sum(self.X), self.X)
        self.assertIs(self.X.sum(zero_var), self.X)


class TestSumGeom(unittest.TestCase):
    """sum_geom(p): sum of a Geometric(p)-count of iid copies of X."""

    def setUp(self):
        self.X = geometric(0.5)
        self.p = 0.4

    def test_mean_matches_wald_identity(self):
        # E[sum of N iid X_i] = E[N]*E[X], Wald's identity; E[N] = 1/p here.
        res = self.X.sum_geom(self.p)
        self.assertAlmostEqual(res.mean(), (1 / self.p) * self.X.mean(), places=8)

    def test_matches_brute_force_compound_pmf(self):
        res = self.X.sum_geom(self.p)
        pmf_x = self.X.pmf_range(40)
        kmax = 40
        brute = np.zeros(kmax + 1)
        conv_n = np.array([1.0])
        pN = self.p
        for n in range(1, 200):
            conv_n = np.convolve(conv_n, pmf_x)[:kmax + 1]
            pN_n = self.p * (1 - self.p) ** (n - 1)
            brute[:len(conv_n)] += pN_n * conv_n[:kmax + 1]
            if pN_n < 1e-14:
                break
        assert_allclose(res.pmf_range(kmax), brute, atol=1e-6)


class TestSumPH(unittest.TestCase):
    """
    sum_ph(counter): sum of a DPH-distributed number of iid copies of X.
    Validated two ways: against sum_geom (its 1-phase special case) and
    against a brute-force compound sum built from counter's own pmf.
    """

    def setUp(self):
        self.X = geometric(0.5)

    def test_reduces_to_self_when_counter_is_deterministic_one(self):
        counter_one = D(np.array([1.0]), np.array([[0.0]]))
        res = self.X.sum_ph(counter_one)
        assert_allclose(res.alpha, self.X.alpha, atol=1e-10)
        assert_allclose(res.A, self.X.A, atol=1e-10)

    def test_matches_sum_geom_for_one_phase_counter(self):
        p = 0.4
        counter = D(np.array([1.0]), np.array([[1 - p]]))  # Geometric(p) count
        via_sum_ph = self.X.sum_ph(counter)
        via_sum_geom = self.X.sum_geom(p)
        assert_allclose(via_sum_ph.pmf_range(30), via_sum_geom.pmf_range(30),
                         atol=1e-8)
        self.assertAlmostEqual(via_sum_ph.mean(), via_sum_geom.mean(), places=8)

    def test_matches_brute_force_compound_sum_multiphase_counter(self):
        counter = D(np.array([0.6, 0.4]), np.array([[0.1, 0.2], [0.0, 0.3]]))
        res = self.X.sum_ph(counter)

        kmax = 25
        pN = counter.pmf_range(40)
        brute = np.zeros(kmax + 1)
        brute[0] += pN[0]
        conv_n = np.array([1.0])
        pmf_x = self.X.pmf_range(kmax)
        for n in range(1, len(pN)):
            conv_n = np.convolve(conv_n, pmf_x)[:kmax + 1]
            brute[:len(conv_n)] += pN[n] * conv_n[:kmax + 1]
        assert_allclose(res.pmf_range(kmax), brute, atol=1e-6)


class TestMix(unittest.TestCase):
    """mix(p, other): the resulting pmf must be the literal p-weighted mix."""

    def setUp(self):
        self.X = geometric(0.5)
        self.Y = negative_binomial(2, 0.4)
        self.p = 0.35

    def test_pmf_is_linear_combination(self):
        m = self.X.mix(self.p, self.Y)
        expected = self.p * self.X.pmf_range(20) + (1 - self.p) * self.Y.pmf_range(20)
        assert_allclose(m.pmf_range(20), expected, atol=1e-10)

    def test_mean_is_linear_combination(self):
        m = self.X.mix(self.p, self.Y)
        expected = self.p * self.X.mean() + (1 - self.p) * self.Y.mean()
        self.assertAlmostEqual(m.mean(), expected, places=8)

    def test_phase_count(self):
        m = self.X.mix(self.p, self.Y)
        self.assertEqual(m.n_phases, self.X.n_phases + self.Y.n_phases)

    def test_rejects_p_out_of_range(self):
        with self.assertRaises(ValueError):
            self.X.mix(1.5, self.Y)


class TestClosureOperationsDenseSparse(unittest.TestCase):
    """
    Closure operations must give numerically identical distributions
    regardless of whether the operands are stored dense or sparse, and must
    return an object whose OWN storage matches ``self`` (not the operand) -
    e.g. a Dense variable combined with a Sparse one still returns Dense.
    """

    def setUp(self):
        self.Xd = geometric(0.5)
        self.Yd = geometric(0.7)
        self.Ys = S(np.array([1.0]), sp.csr_array([[0.3]]))  # same as Yd

    def test_min_matches_across_storage(self):
        mn_dense = self.Xd.min(self.Yd)
        mn_mixed = self.Xd.min(self.Ys)
        assert_allclose(mn_dense.A, mn_mixed.A, atol=1e-10)
        assert_allclose(mn_dense.alpha, mn_mixed.alpha, atol=1e-10)

    def test_result_storage_follows_self_not_other(self):
        self.assertFalse(sp.issparse(self.Xd.min(self.Ys).A))
        self.assertTrue(sp.issparse(self.Ys.min(self.Xd).A))

    def test_sum_and_max_match_across_storage(self):
        for op in ("sum", "max", "mix"):
            if op == "mix":
                a = getattr(self.Xd, op)(0.5, self.Yd)
                b = getattr(self.Xd, op)(0.5, self.Ys)
            else:
                a = getattr(self.Xd, op)(self.Yd)
                b = getattr(self.Xd, op)(self.Ys)
            assert_allclose(a.pmf_range(15), b.pmf_range(15), atol=1e-8,
                             err_msg=f"mismatch for {op}")


if __name__ == "__main__":
    unittest.main()
