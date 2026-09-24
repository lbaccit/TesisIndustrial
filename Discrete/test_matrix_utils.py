"""
test_matrix_utils.py

Tests for matrix_utils.py.

Coverage:
  1. Equivalence with the MatrixUtils.java loops, transcribed literally as
     reference functions inside this file (Java cannot be run here, so the
     "Java truth" is reproduced by hand).
  2. Numerical equivalence between dense input (np.ndarray) and sparse input
     (scipy.sparse), across both scipy APIs (csr_matrix and csr_array).
  3. The three documented Java bugs, checking that they are NOT reproduced here.
  4. Edge cases (k=0, 1x1 matrices, validations that must fail).

There are no official Java tests for MatrixUtils (jMarkov ships no
MatrixUtilsTest.java), so the J_* reference functions in this file are the only
external source of truth available. They are written so that anyone can read
them line by line against the mathematical definition, without using the NumPy
shortcuts that are the thing under test.

Run with:
    python3 -m unittest test_matrix_utils -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

import matrix_utils as mu


# =============================================================================
# Literal transcriptions of the MatrixUtils.java loops (reference)
# =============================================================================

def J_kron_mm(A, B):
    """kronecker(Matrix A, Matrix B, Matrix res), transcribed loop by loop."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    r1, c1 = A.shape
    r2, c2 = B.shape
    R = np.zeros((r1 * r2, c1 * c2))
    for i1 in range(r1):
        for j1 in range(c1):
            for i2 in range(r2):
                for j2 in range(c2):
                    R[i1 * r2 + i2, j1 * c2 + j2] = A[i1, j1] * B[i2, j2]
    return R


def J_kron_mx_col_vector(A, b):
    """kronecker(Matrix A, Vector B, Matrix res), with b as a column."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    r1, c1 = A.shape
    n2 = len(b)
    R = np.zeros((r1 * n2, c1))
    for i1 in range(r1):
        for j1 in range(c1):
            for i2 in range(n2):
                R[i1 * n2 + i2, j1] = A[i1, j1] * b[i2]
    return R


def J_kron_col_vector_mx(a, B):
    """kronecker(Vector A, Matrix B, Matrix res), with a as a column."""
    a = np.asarray(a, dtype=float)
    B = np.asarray(B, dtype=float)
    n1 = len(a)
    r2, c2 = B.shape
    R = np.zeros((n1 * r2, c2))
    for i1 in range(n1):
        for i2 in range(r2):
            for j2 in range(c2):
                R[i1 * r2 + i2, j2] = a[i1] * B[i2, j2]
    return R


def J_concat_quad(lu, ru, ld, rd):
    """concatQuad: block matrix assembly, loop by loop."""
    lu, ru, ld, rd = (np.asarray(x, dtype=float) for x in (lu, ru, ld, rd))
    rows_top, cols_left = lu.shape
    rows_bottom, cols_right = rd.shape
    R = np.zeros((rows_top + rows_bottom, cols_left + cols_right))
    R[:rows_top, :cols_left] = lu
    R[:rows_top, cols_left:] = ru
    R[rows_top:, :cols_left] = ld
    R[rows_top:, cols_left:] = rd
    return R


def J_mat_power(A, k):
    """matPower(Matrix A, int k): repeated multiplication, no shortcuts."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    R = np.eye(n)
    for _ in range(k):
        R = R @ A
    return R


def J_sum_mat_power_correct(A, k):
    """
    The sum I + A + A^2 + ... + A^(k-1), which is what sumMatPower SHOULD
    compute. Used as the mathematical reference, not as a transcription of the
    Java bug (that one is tested separately, in test_sum_mat_power_java_bug).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    total = np.eye(n)
    term = np.eye(n)
    for _ in range(1, k):
        term = term @ A
        total = total + term
    return total


# =============================================================================
# Shared fixtures
# =============================================================================

# 3x3 sub-generator matrix (continuous case), with strict absorption.
G3 = np.array([[-2.0, 2.0, 0.0],
               [0.0, -3.0, 1.0],
               [0.0, 0.0, -1.0]])

# 3x3 sub-stochastic matrix (discrete case), with absorption.
D3 = np.array([[0.5, 0.2, 0.0],
               [0.1, 0.6, 0.1],
               [0.0, 0.0, 0.4]])

V3 = np.array([0.5, 0.3, 0.2])
W3 = np.array([1.0, 2.0, 3.0])


def _is_matrix_like(x):
    return mu.is_sparse(x) or (hasattr(x, "ndim") and x.ndim == 2)


class MatrixUtilsTestCase(unittest.TestCase):
    """Base class with a helper to compare dense against both scipy APIs."""

    def assert_dense_sparse_equal(self, fn, *args, tol=1e-11):
        """
        Call ``fn`` with the dense ``args``, then again converting every matrix
        argument (ndim == 2) to csr_matrix and to csr_array. All three outputs
        must agree once densified.
        """
        dense = mu.to_dense(fn(*args)) if _is_matrix_like(fn(*args)) \
            else np.asarray(fn(*args), dtype=float)

        for api in (sp.csr_matrix, sp.csr_array):
            sparse_args = [api(a) if (np.ndim(a) == 2) else a for a in args]
            result = fn(*sparse_args)
            result = mu.to_dense(result) if mu.is_sparse(result) \
                else np.asarray(result, dtype=float)
            assert_allclose(result, dense, atol=tol,
                            err_msg=f"{fn.__name__} with {api.__name__}")


# =============================================================================
# Part 1: functions ported from MatrixUtils.java
# =============================================================================

class TestOnesHelpers(unittest.TestCase):
    def test_ones_vector(self):
        assert_allclose(mu.ones_vector(3), [1.0, 1.0, 1.0])

    def test_ones_row_col(self):
        self.assertEqual(mu.ones_row(3).shape, (1, 3))
        self.assertEqual(mu.ones_col(3).shape, (3, 1))


class TestKronecker(MatrixUtilsTestCase):
    def test_kronecker_matches_java_cycle(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        assert_allclose(mu.kronecker(A, B), J_kron_mm(A, B))

    def test_kronecker_matches_numpy(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        assert_allclose(mu.kronecker(A, B), np.kron(A, B))

    def test_kronecker_mx_col_vector_matches_java_cycle(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        assert_allclose(mu.kronecker_mx_col_vector(A, b),
                        J_kron_mx_col_vector(A, b))

    def test_kronecker_col_vector_mx_matches_java_cycle(self):
        a = np.array([1.0, 2.0])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        assert_allclose(mu.kronecker_col_vector_mx(a, B),
                        J_kron_col_vector_mx(a, B))

    def test_kronecker_sum_shape_and_formula(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        ks = mu.kronecker_sum(A, B)
        expected = np.kron(A, np.eye(2)) + np.kron(np.eye(2), B)
        self.assertEqual(ks.shape, (4, 4))
        assert_allclose(ks, expected)

    def test_kronecker_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        assert_allclose(mu.kronecker_vectors(a, b), np.kron(a, b))

    def test_dense_vs_sparse(self):
        self.assert_dense_sparse_equal(mu.kronecker, D3, D3)
        self.assert_dense_sparse_equal(mu.kronecker_sum, D3, D3)
        self.assert_dense_sparse_equal(mu.kronecker_mx_col_vector, D3, W3)
        self.assert_dense_sparse_equal(mu.kronecker_col_vector_mx, W3, D3)
        self.assert_dense_sparse_equal(mu.kronecker_mx_row_vector, D3, W3)


class TestConcat(MatrixUtilsTestCase):
    def test_concat_rows_cols(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        self.assertEqual(mu.concat_rows(A, B).shape, (4, 2))
        self.assertEqual(mu.concat_cols(A, B).shape, (2, 4))

    def test_concat_rows_cols_incompatible_raises(self):
        A = np.zeros((2, 2))
        B = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            mu.concat_rows(A, B)
        with self.assertRaises(ValueError):
            mu.concat_cols(np.zeros((2, 2)), np.zeros((3, 2)))

    def test_concat_quad_matches_java_cycle(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        result = mu.concat_quad(A, B, B, A)
        assert_allclose(result, J_concat_quad(A, B, B, A))

    def test_concat_quad_blocks_placed_correctly(self):
        q = mu.concat_quad(D3, D3 * 0, D3 * 0, D3)
        self.assertEqual(q.shape, (6, 6))
        assert_allclose(q[:3, :3], D3)
        assert_allclose(q[3:, 3:], D3)
        assert_allclose(q[:3, 3:], np.zeros((3, 3)))

    def test_concat_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        assert_allclose(mu.concat_vectors(a, b), [1, 2, 3, 4, 5])

    def test_dense_vs_sparse(self):
        self.assert_dense_sparse_equal(mu.concat_rows, D3, D3)
        self.assert_dense_sparse_equal(mu.concat_cols, D3, D3)
        self.assert_dense_sparse_equal(mu.concat_quad, D3, D3, D3, D3)


class TestMultVector(unittest.TestCase):
    def test_mult_vector_is_outer_product(self):
        a = np.array([1.0, 2.0])
        assert_allclose(mu.mult_vector(a, a), np.outer(a, a))


class TestMatPower(MatrixUtilsTestCase):
    def test_mat_power_matches_java_cycle(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        for k in (0, 1, 3, 5):
            assert_allclose(mu.mat_power(A, k), J_mat_power(A, k))

    def test_mat_power_k0_is_identity(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(mu.mat_power(A, 0), np.eye(2))

    def test_mat_power_with_vectors(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = np.array([1.0, 2.0])
        expected = a @ (A @ A @ A) @ a
        self.assertAlmostEqual(mu.mat_power(A, 3, a, a), expected, places=10)

    def test_mat_power_rejects_negative_k(self):
        with self.assertRaises(ValueError):
            mu.mat_power(D3, -1)

    def test_mat_power_rejects_one_sided_vectors(self):
        with self.assertRaises(ValueError):
            mu.mat_power(D3, 2, left_vec=V3)  # right_vec is missing

    def test_dense_vs_sparse(self):
        for k in (0, 1, 2, 5):
            self.assert_dense_sparse_equal(mu.mat_power, D3, k)

    def test_dense_vs_sparse_csr_array_power_bug_regression(self):
        """
        Explicit regression for the finding: on csr_array, A**k is element-wise
        power, not matrix power. mat_power must give the MATRIX power in both
        cases.
        """
        A = sp.csr_array(D3)
        result = mu.to_dense(mu.mat_power(A, 3))
        assert_allclose(result, np.linalg.matrix_power(D3, 3), atol=1e-11)
        # What must NOT happen: matching the element-wise power.
        element_wise = mu.to_dense(A) ** 3
        self.assertFalse(np.allclose(result, element_wise))


class TestSumMatPower(unittest.TestCase):
    def test_sum_mat_power_matches_formula(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        for k in (1, 3, 5):
            assert_allclose(mu.sum_mat_power(A, k), J_sum_mat_power_correct(A, k))

    def test_sum_mat_power_with_vectors(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = np.array([1.0, 2.0])
        total = J_sum_mat_power_correct(A, 4)
        expected = a @ (total @ a)
        self.assertAlmostEqual(mu.sum_mat_power(A, 4, a, a), expected, places=10)

    def test_sum_mat_power_java_bug(self):
        """
        Documented in the module: sumMatPower in Java does not accumulate the
        powers and returns k*I. Here the real sum (I+A+A^2+...) must differ
        from k*I except in degenerate cases.
        """
        A = np.array([[0.5, 0.2], [0.1, 0.6]])
        k = 4
        result = mu.sum_mat_power(A, k)
        java_bug = k * np.eye(2)
        self.assertFalse(np.allclose(result, java_bug),
                         "sum_mat_power must not reproduce the Java bug")

    def test_sum_mat_power_rejects_k_less_than_1(self):
        with self.assertRaises(ValueError):
            mu.sum_mat_power(D3, 0)


class TestSolvePower(unittest.TestCase):
    def test_solve_power_k0_returns_v0(self):
        v0 = np.array([1.0, 2.0])
        assert_allclose(mu.solve_power(np.eye(2), 0, v0), v0)

    def test_solve_power_matches_repeated_solve(self):
        M = np.array([[2.0, 0.0], [0.0, 4.0]])
        # (M^-1)^3 * 1 = 1 / (2^3, 4^3)
        result = mu.solve_power(M, 3)
        expected = np.array([1 / 8, 1 / 64])
        assert_allclose(result, expected)

    def test_solve_power_dense_vs_sparse(self):
        M = np.eye(3) - D3
        dense = mu.solve_power(M, 3)
        for api in (sp.csc_matrix, sp.csc_array):
            sparse_result = mu.solve_power(api(M), 3)
            assert_allclose(sparse_result, dense, atol=1e-10)

    def test_solve_power_rejects_negative_k(self):
        with self.assertRaises(ValueError):
            mu.solve_power(np.eye(2), -1)


class TestDistanceScalarStats(unittest.TestCase):
    def test_distance_relative(self):
        v1 = np.array([2.0, 4.0])
        v2 = np.array([1.0, 4.0])
        self.assertAlmostEqual(mu.distance(v1, v2), 0.5, places=10)

    def test_scalar_extracts_1x1(self):
        self.assertEqual(mu.scalar(np.array([[7.0]])), 7.0)

    def test_average_and_variance_population(self):
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(mu.average(data), 2.5, places=10)
        self.assertAlmostEqual(mu.variance(data), 1.25, places=10)

    def test_cv_is_scv_like_java(self):
        """
        Java: CV(double[]) actually returns Var/mean^2 (the SCV), not the
        relative standard deviation. The name is kept for fidelity, and
        cv_true gives the real CV.
        """
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(mu.cv(data), 1.25 / 6.25, places=10)
        self.assertAlmostEqual(mu.cv_true(data), np.sqrt(1.25) / 2.5, places=10)


class TestValidations(unittest.TestCase):
    def test_check_sub_stochastic_vector(self):
        self.assertTrue(mu.check_sub_stochastic_vector([0.4, 0.3]))
        self.assertFalse(mu.check_sub_stochastic_vector([0.7, 0.6]))
        self.assertFalse(mu.check_sub_stochastic_vector([1.2, -0.2]))

    def test_check_sub_generator_matrix_accepts_valid(self):
        self.assertTrue(mu.check_sub_generator_matrix(G3))

    def test_check_sub_generator_matrix_java_gaps_closed(self):
        """
        Documented in matrix_utils.py: Java's checkSubGeneratorMatrix (a) never
        inspects column 0, and (b) accepts a full generator (with no exit to
        absorption) because it does not require some row to sum strictly below
        zero. This implementation closes both gaps.
        """
        full_generator = np.array([[-1.0, 1.0], [1.0, -1.0]])
        self.assertFalse(mu.check_sub_generator_matrix(full_generator))

        positive_diagonal_col0 = np.array([[1.0, -1.0], [0.0, -1.0]])
        self.assertFalse(mu.check_sub_generator_matrix(positive_diagonal_col0))

    def test_check_sub_stochastic_matrix(self):
        self.assertTrue(mu.check_sub_stochastic_matrix(D3))
        recurrent_matrix = np.array([[1.0, 0.0], [0.5, 0.4]])
        self.assertFalse(mu.check_sub_stochastic_matrix(recurrent_matrix))


class TestVec0Mat0(unittest.TestCase):
    def test_vec0(self):
        alpha = np.array([0.4, 0.3])
        self.assertAlmostEqual(mu.vec0(alpha), 0.3, places=10)

    def test_mat0_discrete(self):
        assert_allclose(mu.mat0(D3, discrete=True), np.ones(3) - D3 @ np.ones(3))

    def test_mat0_continuous(self):
        assert_allclose(mu.mat0(G3, discrete=False), -(G3 @ np.ones(3)))

    def test_dense_vs_sparse(self):
        for api in (sp.csr_matrix, sp.csr_array):
            assert_allclose(mu.to_dense(mu.mat0(api(D3), True)),
                            mu.mat0(D3, True), atol=1e-11)
            assert_allclose(mu.to_dense(mu.mat0(api(G3), False)),
                            mu.mat0(G3, False), atol=1e-11)


# =============================================================================
# Part 3: sparse dispatch infrastructure (is_sparse, eye_like, to_dense)
# =============================================================================

class TestSparseDispatchInfra(unittest.TestCase):
    def test_is_sparse(self):
        self.assertFalse(mu.is_sparse(D3))
        self.assertTrue(mu.is_sparse(sp.csr_matrix(D3)))
        self.assertTrue(mu.is_sparse(sp.csr_array(D3)))

    def test_eye_like_dense(self):
        assert_allclose(mu.eye_like(D3), np.eye(3))

    def test_eye_like_sparse_is_actually_sparse_and_correct(self):
        for api in (sp.csr_matrix, sp.csr_array):
            identity = mu.eye_like(api(D3))
            self.assertTrue(mu.is_sparse(identity))
            assert_allclose(mu.to_dense(identity), np.eye(3))

    def test_to_dense_noop_on_dense(self):
        assert_allclose(mu.to_dense(D3), D3)

    def test_to_dense_on_sparse(self):
        assert_allclose(mu.to_dense(sp.csr_array(D3)), D3)


if __name__ == "__main__":
    unittest.main()
