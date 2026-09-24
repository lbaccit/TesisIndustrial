"""
jmarkov/phase/matrix_utils.py

Matrix utilities for the Phase-Type package, shared by the continuous (ctph)
and discrete (dtph) cases.

This is the counterpart of ``jphase/MatrixUtils.java`` in the original project:

    jMarkov/src/jphase/MatrixUtils.java   ->   jmarkov/phase/matrix_utils.py

Names are kept one to one, converted from camelCase to snake_case (PEP 8). Each
function states in its docstring which Java method it comes from.

Watch out for the import: there is also a ``jmarkov/matrix_utils.py`` at the
root of the package (general purpose, it holds ``exp_unif``). They are two
different modules:

    from jmarkov.matrix_utils import exp_unif          # general purpose
    from jmarkov.phase.matrix_utils import mat_power   # this one

About the Java overloads
------------------------
Many Java methods appear two or three times with different signatures. Most of
those variants exist only because MTJ works with pre-allocated output matrices
(the ``res`` parameter), not because they compute anything different. In NumPy
that need disappears: ``kronecker(A, B, res)`` and ``kronecker(A, B)`` collapse
into a single function. When two overloads really do differ (``matPower``, for
instance), they are resolved with optional arguments.

Not ported from MatrixUtils.java
--------------------------------
- ``exp``, ``expUnif``, ``expTimesOnes``: uniformization already lives in
  ``jmarkov/matrix_utils.py`` as ``exp_unif``, together with its private
  helpers ``computeLdaMax`` and ``resultFromMedian``. For the direct matrix
  exponential we use ``scipy.linalg.expm``, which implements the Al-Mohy &
  Higham (2009) algorithm, the same one cited in the comment in ``ctph.py``.
- ``expRunge`` and its private helper ``runge4``: a fourth-order Runge-Kutta
  integrator offered as an alternative to uniformization. Pending, in case we
  want to reproduce Java's ``useUniformization = false`` flag.
- ``pow(x, n)``: in Python this is ``x ** n``.
- Overloads taking a pre-allocated output matrix (``kronecker(..., res)``,
  ``OnesVector(Vector vec)``, ``concatRows(..., res)``, and so on): they exist
  only because of how MTJ works and add nothing in NumPy.

Dense vs. sparse
----------------
Functions that take matrices accept both dense ``np.ndarray`` and
``scipy.sparse`` matrices, and dispatch internally. This mirrors what Java
does: there, ``MatrixUtils`` operates on MTJ's ``Matrix`` interface, which both
``DenseMatrix`` and ``FlexCompRowMatrix`` implement, which is why
``SparseContPhaseVar`` and ``SparseDiscPhaseVar`` can reuse it. Python has no
such common interface, so the dispatch is made explicit through ``is_sparse``.

The ``dtype=float`` that appears throughout the code fixes the ELEMENT TYPE
(float64 instead of int), not the density. What forces density is
``np.asarray``, and for that reason it is only used on the dense branch.

VECTORS (alpha, the output vector) are always densified. This is a deliberate
choice: they cost O(n) against the O(n^2) of the matrix, so the saving would be
marginal while complicating every downstream operation.

What is at stake: a bidiagonal sub-generator with n = 10,000 takes 800 MB dense
against 0.3 MB in CSR, a factor of 2,857x.

Three traps that motivate the explicit dispatch, because they do NOT fail loudly
--------------------------------------------------------------------------------
1. ``np.kron(sparse, sparse)`` returns a SILENTLY WRONG result: for two 2x2
   matrices it hands back a 2x2 instead of the 4x4, that is, an element-wise
   product by broadcasting rather than a Kronecker product.
2. ``np.block([[sparse, ...]])`` returns a 2x2 array of sparse objects with
   ``dtype=object``, not the assembled matrix.
3. ``A ** k`` changes meaning between the two scipy APIs: on ``spmatrix``
   (csr_matrix) it is MATRIX power, while on ``sparray`` (csr_array) it is
   ELEMENT-WISE power, just as on ndarray. With csr_array, ``A ** 2`` returns
   the squared entries and gives no warning. That is why ``mat_power``
   multiplies explicitly on the sparse branch.

Operation correspondence
------------------------
    dense                            sparse
    ------------------------------   ----------------------------------
    A @ np.ones(n)                   A @ np.ones(n)          (same)
    A.sum(axis=1)                    A.sum(axis=1)           (same)
    np.eye(n)                        eye_like(A)   <- NOT A ** 0
    np.linalg.matrix_power(A, k)     explicit multiplication
    np.linalg.solve(M, v)            splu(M).solve(v), factorizing once
    np.linalg.eigvals(A)             scipy.sparse.linalg.eigs(A, k=1)
    np.kron(A, B)                    scipy.sparse.kron(A, B)
    np.block([[...]])                scipy.sparse.bmat([[...]])
    scipy.linalg.expm(A)             scipy.sparse.linalg.expm(A)


Three Java methods contain bugs that are NOT reproduced here. Each one is
explained in the docstring of the corresponding function:

1. ``sumMatPower`` - does not accumulate the powers; it returns k*I. See
   ``sum_mat_power``.
2. ``kroneckerMxRowVector`` - wrong column stride. See
   ``kronecker_mx_row_vector``.
3. ``checkSubGeneratorMatrix`` - never inspects column 0 and does not require
   an exit to absorption. See ``check_sub_generator_matrix``.

All three are dead code inside jphase (nobody calls 1 and 2; 3 is used, but
loosely), so fixing them should not break anything. Even so, it is worth
confirming with the advisor before proposing changes to the Java side.

There are also two deliberate behavioural differences:

- ``sumMatPower(k < 1)`` in Java prints a message and returns 0;
  ``sum_mat_power`` raises ``ValueError``.
- ``CV(double[])`` in Java returns Var/mean^2, which is the SCV and not the CV.
  The name and the behaviour are kept in ``cv``, and ``cv_true`` is added.

References
----------
[1] Neuts, M. F. (1981). Matrix-Geometric Solutions in Stochastic Models.
[2] Latouche, G. & Ramaswami, V. (1999). Introduction to Matrix Analytic
    Methods in Stochastic Modeling.
[3] Al-Mohy, A. H. & Higham, N. J. (2009). SIAM J. Matrix Anal. Appl. 31(3).

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

from math import comb, factorial, sqrt
from typing import Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

# Numerical tolerance. Equivalent to the `Epsilon` constant in MatrixUtils.java.
EPS = 1.0e-10


# =============================================================================
# PART 1 - Ported from jphase.MatrixUtils
# =============================================================================

def ones_vector(m: int) -> np.ndarray:
    """1-D vector of ones. Java: ``OnesVector(int m)``."""
    return np.ones(m, dtype=float)


def ones_row(m: int) -> np.ndarray:
    """Row matrix (1, m) of ones. Java: ``OnesRow(int m)``."""
    return np.ones((1, m), dtype=float)


def ones_col(m: int) -> np.ndarray:
    """Column matrix (m, 1) of ones. Java: ``OnesCol(int m)``."""
    return np.ones((m, 1), dtype=float)


def kronecker(A, B) -> np.ndarray:
    """
    Kronecker product A (x) B. Java: ``kronecker(Matrix A, Matrix B)``.

    Used by the PH closure operations (min, max).

    Java has four ``kronecker`` overloads because it distinguishes the Matrix
    and Vector types, and because a Vector can enter either as a column or as a
    row. In NumPy everything is an ndarray, so here they are split by name:

        kronecker(Matrix A, Matrix B, res)       -> kronecker(A, B)
        kronecker(Matrix A, Vector B, res)       -> kronecker_mx_col_vector(A, b)
        kronecker(Vector A, Matrix B, res)       -> kronecker_col_vector_mx(a, B)
        kroneckerMxRowVector(Matrix A, Vector B) -> kronecker_mx_row_vector(A, b)
    """
    if is_sparse(A) or is_sparse(B):
        return sparse.kron(A, B, format="csr")
    return np.kron(np.asarray(A, dtype=float), np.asarray(B, dtype=float))


def kronecker_mx_col_vector(A, b) -> np.ndarray:
    """
    A (x) b, with b treated as a COLUMN vector.

    Java: ``kronecker(Matrix A, Vector B, Matrix res)``, whose output matrix has
    size (A.numRows * b.size, A.numColumns).
    """
    b = as_vector(b, "b").reshape(-1, 1)
    if is_sparse(A):
        return sparse.kron(A, b, format="csr")
    return np.kron(np.asarray(A, dtype=float), b)


def kronecker_col_vector_mx(a, B) -> np.ndarray:
    """
    a (x) B, with a treated as a COLUMN vector.

    Java: ``kronecker(Vector A, Matrix B, Matrix res)``, whose output matrix has
    size (a.size * B.numRows, B.numColumns).
    """
    a = as_vector(a, "a").reshape(-1, 1)
    if is_sparse(B):
        return sparse.kron(a, B, format="csr")
    return np.kron(a, np.asarray(B, dtype=float))


def kronecker_mx_row_vector(A, b) -> np.ndarray:
    """
    A (x) b, with b treated as a ROW vector.

    Java: ``kroneckerMxRowVector(Matrix A, Vector B, Matrix res)``, with output
    of size (A.numRows, A.numColumns * b.size).

    Divergence from Java
    --------------------
    The Java version writes into column ``j1 * c1 + i2``, where ``c1`` is the
    number of columns of A. The correct stride is ``j1 * n2 + i2``, with
    ``n2 = b.size``. The two agree only when A has as many columns as b has
    entries. With A of size 2x3 and b of size 2, for example, Java writes into
    columns {0,1,3,4,6,7} of a 2x6 matrix: it runs out of range and leaves
    columns 2 and 5 unfilled.

    The correct Kronecker product is implemented here. Nothing else in jphase
    calls this method, so the bug never showed up.
    """
    b = as_vector(b, "b").reshape(1, -1)
    if is_sparse(A):
        return sparse.kron(A, b, format="csr")
    return np.kron(np.asarray(A, dtype=float), b)


def kronecker_sum(A, B) -> np.ndarray:
    """
    Kronecker sum A (+) B = A (x) I_nb + I_na (x) B.

    Java: ``kroneckerSum(Matrix A, Matrix B)``.

    It is the generator of two processes evolving in parallel; it shows up in
    the minimum and the maximum of two PH variables.
    """
    return (kronecker(A, eye_like(B))
            + kronecker(eye_like(A), B))


def kronecker_vectors(a, b) -> np.ndarray:
    """Kronecker product of two vectors. Java: ``kroneckerVectors``."""
    return np.kron(as_vector(a, "a"), as_vector(b, "b"))


def concat_rows(A, B) -> np.ndarray:
    """
    Stack A on top of B (same number of columns). Java: ``concatRows``.
    """
    if is_sparse(A) or is_sparse(B):
        return sparse.vstack([A, B], format="csr")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"concat_rows: incompatible columns {A.shape[1]} != {B.shape[1]}."
        )
    return np.vstack((A, B))


def concat_cols(A, B) -> np.ndarray:
    """
    Place A and B side by side (same number of rows). Java: ``concatCols``.
    """
    if is_sparse(A) or is_sparse(B):
        return sparse.hstack([A, B], format="csr")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"concat_cols: incompatible rows {A.shape[0]} != {B.shape[0]}."
        )
    return np.hstack((A, B))


def concat_quad(left_up, right_up, left_down, right_down) -> np.ndarray:
    """
    Assemble the block matrix

        [ left_up    right_up   ]
        [ left_down  right_down ]

    Java: ``concatQuad(leftUp, rightUp, leftDown, rightDown, res)``.

    This is the assembly used by sum(), mix(), min() and max() to build the
    generator of the resulting variable.
    """
    blocks = (left_up, right_up, left_down, right_down)
    if any(is_sparse(b) for b in blocks):
        return sparse.bmat([[left_up, right_up], [left_down, right_down]],
                           format="csr")
    lu = np.atleast_2d(np.asarray(left_up, dtype=float))
    ru = np.atleast_2d(np.asarray(right_up, dtype=float))
    ld = np.atleast_2d(np.asarray(left_down, dtype=float))
    rd = np.atleast_2d(np.asarray(right_down, dtype=float))
    if lu.shape[0] != ru.shape[0] or ld.shape[0] != rd.shape[0]:
        raise ValueError("concat_quad: incompatible rows between blocks.")
    if lu.shape[1] != ld.shape[1] or ru.shape[1] != rd.shape[1]:
        raise ValueError("concat_quad: incompatible columns between blocks.")
    return np.block([[lu, ru], [ld, rd]])


def concat_vectors(a, b) -> np.ndarray:
    """Concatenate two 1-D vectors. Java: ``concatVectors``."""
    return np.concatenate((np.asarray(a, dtype=float).ravel(),
                           np.asarray(b, dtype=float).ravel()))


def mult_vector(a, b) -> np.ndarray:
    """
    Outer product a * b^T. Java: ``multVector(Vector A, Vector B, Matrix res)``.

    It appears in the closure operations when connecting the phases of one
    variable to those of the next: ``mult_vector(a1, alpha2)``.
    """
    return np.outer(as_vector(a, "a"), as_vector(b, "b"))


def mat_power(A, k: int, left_vec=None, right_vec=None):
    """
    Matrix power A^k, optionally pre/post-multiplied by vectors.

    Java: covers both ``matPower`` overloads:
      - ``matPower(Matrix A, int k)``                        -> A^k
      - ``matPower(Matrix A, int k, Vector l, Vector r)``    -> l * A^k * r

    Parameters
    ----------
    A : array_like
        Square base matrix.
    k : int
        Exponent, k >= 0. A^0 = I.
    left_vec, right_vec : array_like, optional
        If both are given, returns the scalar ``left_vec @ A^k @ right_vec``
        without forming the full A^k when k is small.

    Returns
    -------
    np.ndarray or float
    """
    if k < 0:
        raise ValueError(f"mat_power: k must be >= 0; got {k}.")
    if (left_vec is None) != (right_vec is None):
        raise ValueError(
            "mat_power: 'left_vec' and 'right_vec' must be given together, or "
            "neither of them."
        )
    if is_sparse(A):
        # ``A ** k`` is NOT used: the operator changes meaning between the two
        # scipy APIs. On spmatrix (csr_matrix) it is MATRIX power; on sparray
        # (csr_array) it is ELEMENT-WISE power, just as on ndarray. With
        # csr_array, ``A ** 2`` returns the squared entries, not A@A, and gives
        # no warning. So we multiply explicitly.
        Ak = eye_like(A)
        for _ in range(k):
            Ak = Ak @ A
    else:
        A = np.asarray(A, dtype=float)
        Ak = np.linalg.matrix_power(A, k)
    if left_vec is None:
        return Ak
    left = as_vector(left_vec, "left_vec")
    right = as_vector(right_vec, "right_vec")
    return float(left @ (Ak @ right))


def sum_mat_power(A, k: int, left_vec=None, right_vec=None):
    """
    Partial sum of powers: S = sum_{j=1}^{k} A^{j-1} = I + A + ... + A^{k-1}.

    Java: ``sumMatPower(Matrix A, int k, Vector leftVec, Vector rightVec)``.

    If the vectors are given, returns the scalar ``left_vec @ S @ right_vec``.

    Notes
    -----
    It shows up when accumulating the discrete CDF and in the loss functions.
    When sp(A) < 1 and k is large, the limit is (I - A)^{-1}; in those cases it
    is better to solve the system than to sum (see ``solve_power``).

    Divergence from Java
    --------------------
    The Java version does not accumulate the powers. Its loop reads:

        Matrix temp = Matrices.identity(n);
        Matrix sum  = temp.copy();
        for (int i = 1; i < k; i++) {
            temp.mult(A, temp.copy());   // <-- result discarded
            sum.add(temp);
        }

    In MTJ, ``X.mult(B, C)`` computes C = X*B and returns C, leaving X
    untouched. Since the return value is never assigned, ``temp`` stays equal
    to the identity on every pass and ``sum`` ends up being k*I. In other
    words, Java returns ``k * (leftVec . rightVec)`` instead of the sum of
    powers. Compare with ``matPower``, which does write
    ``result = result.mult(A, A.copy())``.

    The semantics described by the javadoc are implemented here. No other file
    in jphase calls ``sumMatPower``, so the bug is latent and does not affect
    published results; it is still worth reporting.
    """
    if k < 1:
        raise ValueError(f"sum_mat_power: k must be >= 1; got {k}.")
    total = eye_like(A)
    term = eye_like(A)
    for _ in range(1, k):
        term = term @ A
        total = total + term
    if left_vec is None and right_vec is None:
        return total
    if (left_vec is None) != (right_vec is None):
        raise ValueError(
            "sum_mat_power: 'left_vec' and 'right_vec' must be given together, "
            "or neither of them."
        )
    left = as_vector(left_vec, "left_vec")
    right = as_vector(right_vec, "right_vec")
    return float(left @ (total @ right))


def distance(v1, v2) -> float:
    """
    Maximum relative distance between two arrays. Java: ``distance``.

    For each entry it computes (v1_i - v2_i) / v1_i when v1_i > 0, and the
    absolute difference otherwise; it returns the maximum in absolute value.

    Careful
    -------
    The original javadoc describes it as the "maximum euclidean distance", but
    the code implements a RELATIVE distance (it divides by v1_i). The actual
    behaviour is reproduced here, not the description. It is used as the
    stopping criterion in the jPhaseFit fitting algorithms.
    """
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()
    if v1.shape != v2.shape:
        return -1.0  # same behaviour as Java
    delta = v1 - v2
    positive = v1 > 0
    rel = delta.copy()
    np.divide(delta, v1, out=rel, where=positive)
    return float(np.max(np.abs(rel)))


def scalar(A) -> float:
    """
    Extract the single entry of a one-column matrix. Java: ``scalar``.
    """
    A = np.asarray(A, dtype=float)
    A2 = np.atleast_2d(A)
    if A2.shape[1] != 1:
        raise ValueError(
            f"scalar: expected a matrix with 1 column; got shape {A2.shape}."
        )
    return float(A2[0, 0])


def average(data) -> float:
    """Sample mean. Java: ``average(double[] datos)``."""
    return float(np.mean(np.asarray(data, dtype=float)))


def average2(data) -> float:
    """Second (non-central) moment. Java: ``average2(double[] data)``."""
    arr = np.asarray(data, dtype=float)
    return float(np.mean(arr * arr))


def variance(data) -> float:
    """
    POPULATION variance (divides by n). Java: ``variance(double[] data)``.

    Java's divisor (n, not n-1) is reproduced so that cross-validation against
    jphase matches. If the sample variance is needed, use
    ``np.var(data, ddof=1)``.
    """
    m = average(data)
    return average2(data) - m * m


def cv(data) -> float:
    """
    Java: ``CV(double[] data)``, which returns ``variance / mean^2``.

    Careful with the name
    ---------------------
    That is the SQUARED coefficient of variation (SCV), not the CV. The CV is
    ``sqrt(variance) / mean``. Java's name and behaviour are kept for
    traceability; for the true CV use ``cv_true``.
    """
    m = average(data)
    return variance(data) / (m * m)


def cv_true(data) -> float:
    """True coefficient of variation, sd/mean. Does not exist in Java."""
    return sqrt(variance(data)) / average(data)


def check_sub_stochastic_vector(a, tol: float = EPS) -> bool:
    """
    Check that `a` is a sub-stochastic vector: a_i >= 0 and sum(a) <= 1.

    Java: ``checkSubStochasticVector(Vector a)``.

    This is the validation of the initial vector alpha, identical in the
    continuous and the discrete case. The missing mass, 1 - sum(a), is the
    probability of starting already absorbed.
    """
    a = np.asarray(a, dtype=float).ravel()
    if np.any(a < -tol):
        return False
    return bool(a.sum() <= 1.0 + tol)


def check_sub_generator_matrix(A, tol: float = EPS) -> bool:
    """
    Check that `A` is a sub-generator (CONTINUOUS case).

    Java: ``checkSubGeneratorMatrix(Matrix A)``.

    Conditions:
      1. A_ij >= 0 for i != j
      2. A_ii <  0
      3. sum_j A_ij <= 0 for every row
      4. sum_j A_ij <  0 for at least one row

    Differences from the Java version
    ---------------------------------
    The Java version has two gaps that are closed here:

    a) Its inner loop starts at ``j = 1``, so column 0 is never inspected:
       neither A[0][0] < 0 nor A[i][0] >= 0 is verified.
    b) It does not require condition 4, so a full generator (every row summing
       to 0, with no exit to absorption) passes validation even though it does
       not define a PH: the absorption vector a = -A*1 would be identically
       zero and the variable would never terminate.

    It is worth confirming with the advisor before "fixing" the Java side, in
    case some other part of jphase relies on the lax behaviour.
    """
    if is_sparse(A):
        if A.shape[0] != A.shape[1]:
            return False
        diag = A.diagonal()
        if np.any(diag >= -tol):
            return False
        off = A - (sparse.diags_array(diag) if _is_sparray(A)
                   else sparse.diags(diag))
        if off.nnz and off.min() < -tol:
            return False
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        if np.any(row_sums > tol):
            return False
        return bool(np.any(row_sums < -tol))

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    off_diag = A[~np.eye(n, dtype=bool)]
    if off_diag.size and np.any(off_diag < -tol):
        return False
    if np.any(np.diag(A) >= -tol):
        return False
    row_sums = A.sum(axis=1)
    if np.any(row_sums > tol):
        return False
    return bool(np.any(row_sums < -tol))


# =============================================================================
# PART 2 - Ported from the abstract classes of jphase
#
# In Java these two live as instance methods on AbstractContPhaseVar and
# AbstractDiscPhaseVar (getVec0 / getMat0). Since their definition is identical
# in both hierarchies apart from the sign, they are centralized here as
# functions.
# =============================================================================

def vec0(alpha) -> float:
    """
    Initial probability mass on the absorbing state:

        alpha_0 = 1 - alpha * 1

    Java: ``getVec0()`` (identical in the continuous and the discrete
    hierarchy).

    Continuous: produces an atom at t = 0.  Discrete: it is P(X = 0).
    """
    return 1.0 - float(np.asarray(alpha, dtype=float).sum())


def mat0(A, discrete: bool) -> np.ndarray:
    """
    Absorption vector from each phase.

        Continuous:  a = -A * 1        (absorption rates)
        Discrete:    a =  1 - A * 1    (probability of absorbing in one step)

    Java: ``getMat0()`` in ``AbstractContPhaseVar`` and in
    ``AbstractDiscPhaseVar`` respectively; both versions are unified through the
    `discrete` parameter.
    """
    ones = np.ones(A.shape[0])
    outflow = np.asarray(A @ ones, dtype=float).ravel()
    if discrete:
        return ones - outflow
    return -outflow


# =============================================================================
# PART 3 - Additions that do NOT exist in MatrixUtils.java
#
# Everything in this section is new. It is marked apart so that it is clear
# what is migration and what is our own contribution when defending the thesis.
# =============================================================================

def is_sparse(M) -> bool:
    """
    [NEW] Is `M` a SciPy sparse matrix?

    Covers both scipy.sparse APIs: the old one based on ``spmatrix``
    (csr_matrix, ...) and the new one based on ``sparray`` (csr_array, ...).
    """
    return bool(sparse.issparse(M))


def _is_sparray(M) -> bool:
    """Does `M` belong to the new API (sparray) rather than the old (spmatrix)?"""
    kind = getattr(sparse, "sparray", None)
    return kind is not None and isinstance(M, kind)


def eye_like(A, n: int = None):
    """
    [NEW] Identity of the same type and format as `A`.

    Needed because ``A ** 0`` raises ``NotImplementedError`` in scipy: raising a
    sparse matrix to the zeroth power would produce a dense one, and scipy
    refuses to do that silently. Here the sparse identity is built explicitly.
    """
    n = A.shape[0] if n is None else n
    if not is_sparse(A):
        return np.eye(n)
    if _is_sparray(A):
        return sparse.eye_array(n, format="csr")
    return sparse.identity(n, format="csr")


def to_dense(M) -> np.ndarray:
    """[NEW] Densify `M` if it is sparse; if it is already dense, return it as is."""
    if is_sparse(M):
        return np.asarray(M.todense(), dtype=float)
    return np.asarray(M, dtype=float)


def as_vector(v, name: str = "alpha") -> np.ndarray:
    """
    [NEW] Convert `v` into a 1-D float64 array.

    It accepts a 2-D input with a single row or column and flattens it, because
    writing ``np.array([[0.7, 0.3]])`` instead of ``np.array([0.7, 0.3])`` is a
    common mistake and cannot happen in Java (the Vector and Matrix types are
    distinct there, whereas here both are ndarray).

    It rejects SciPy sparse matrices with an explicit message rather than
    letting ``np.asarray`` raise an unhelpful ValueError.
    """
    if is_sparse(v):
        # Vectors are always densified. That is O(n) against the O(n^2) it
        # costs to densify the matrix, so the saving would be marginal while
        # complicating every downstream operation. Java does use SparseVector,
        # but for uniformity of MTJ types, not for memory.
        v = v.todense()
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1-D; got shape {arr.shape}.")
    return arr


def as_square_matrix(M, name: str = "A") -> np.ndarray:
    """
    [NEW] Convert `M` into a square 2-D float64 array.

    It rejects SciPy sparse matrices with an explicit message.
    """
    if is_sparse(M):
        if M.shape[0] != M.shape[1]:
            raise ValueError(
                f"'{name}' must be square; got {M.shape[0]}x{M.shape[1]}."
            )
        return M.tocsr() if hasattr(M, "tocsr") else M
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"'{name}' must be 2-D; got {arr.ndim} dimension(s)."
        )
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"'{name}' must be square; got {arr.shape[0]}x{arr.shape[1]}."
        )
    return arr


def coerce_representation(alpha, A) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    [NEW] Convert and check the dimensional compatibility of (alpha, A).

    Returns
    -------
    (alpha, A, n_phases)
    """
    A_arr = as_square_matrix(A, "A")
    a_arr = as_vector(alpha, "alpha")
    n = A_arr.shape[0]
    if a_arr.shape[0] != n:
        raise ValueError(
            f"'alpha' must have length {n} to match A; "
            f"got length {a_arr.shape[0]}."
        )
    return a_arr, A_arr, n


def _spectral_radius_lt_one(A, tol: float = EPS) -> bool:
    """
    [NEW] Is sp(A) < 1? Decided with the cheapest method that works.

    1. Sufficient condition: sp(A) <= ||A||_inf = maximum row sum. If that sum
       is < 1, no eigenvalue needs to be computed. This covers the usual case
       and costs O(nnz).
    2. If some row sums to exactly 1, the spectrum has to be inspected. For
       dense or small matrices ``np.linalg.eigvals`` is used. For large sparse
       ones ``scipy.sparse.linalg.eigs`` is used, which requires k < n-1 and
       therefore does not work in small dimensions.
    3. If ``eigs`` does not converge, we fall back to power iteration on the
       vector of ones: since A >= 0 with row sums <= 1, the sequence A^k * 1 is
       monotone non-increasing and tends to the probability of never being
       absorbed.
    """
    row_sums = (np.asarray(A.sum(axis=1)).ravel() if is_sparse(A)
                else np.asarray(A, dtype=float).sum(axis=1))
    if row_sums.size and row_sums.max() < 1.0 - tol:
        return True

    n = A.shape[0]
    if not is_sparse(A) or n <= 1000:
        return bool(np.max(np.abs(np.linalg.eigvals(to_dense(A)))) < 1.0 - tol)

    try:
        lam = spla.eigs(A, k=1, which="LM", return_eigenvectors=False,
                        maxiter=5000)
        return bool(np.max(np.abs(lam)) < 1.0 - tol)
    except Exception:
        u = np.ones(n)
        for _ in range(10_000):
            u = A @ u
            if u.max() < 1e-12:
                return True
        return False


def check_sub_stochastic_matrix(A, tol: float = EPS) -> bool:
    """
    [NEW] Check that `A` is sub-stochastic and transient (DISCRETE case).

    It does not exist in MatrixUtils.java: the Java side only provides
    ``checkSubStochasticVector`` (for alpha) and ``checkSubGeneratorMatrix``
    (for the continuous case). The discrete matrix A is not validated in jphase.

    Conditions:
      1. A_ij >= 0
      2. sum_j A_ij <= 1 for every row
      3. sp(A) < 1

    About condition 3
    -----------------
    Conditions 1 and 2 alone are NOT enough. Counterexample:

        A = [[1.0, 0.0],
             [0.5, 0.4]]

    has row sums (1.0, 0.9), so it is sub-stochastic, but phase 0 never reaches
    absorption: sp(A) = 1, (I - A) is singular and E[X] does not exist.
    sp(A) < 1 is equivalent to absorption happening with probability 1 and to
    (I - A)^{-1} = sum_{k>=0} A^k converging.

    The current check in ``dtph.py`` (rows <= 1, at least one < 1) also accepts
    that counterexample.
    """
    if A.shape[0] != A.shape[1] if is_sparse(A) else (
            np.asarray(A).ndim != 2 or np.asarray(A).shape[0] != np.asarray(A).shape[1]):
        return False
    if is_sparse(A):
        if A.nnz and A.min() < -tol:
            return False
        row_sums = np.asarray(A.sum(axis=1)).ravel()
    else:
        A = np.asarray(A, dtype=float)
        if np.any(A < -tol):
            return False
        row_sums = A.sum(axis=1)
    if np.any(row_sums > 1.0 + tol):
        return False
    return _spectral_radius_lt_one(A, tol)


def solve_power(M, k: int, v0=None) -> np.ndarray:
    """
    [NEW] Compute M^{-k} v0 by solving k linear systems.

    It replaces the Java pattern

        TInv = T.solve(I, ...)          // explicit inverse
        TInv = matPower(TInv, k)        // then raised to the k-th power

    which inverts the matrix explicitly. Solving the system propagates less
    rounding error: inverting and multiplying applies the condition number one
    more time than necessary.

    It is used with:
        M = -A,  v0 = 1            -> continuous, (-A)^{-k} 1
        M = I-A, v0 = A^{k-1} 1    -> discrete,  (I-A)^{-k} A^{k-1} 1
    """
    if k < 0:
        raise ValueError(f"solve_power: k must be >= 0; got {k}.")
    n = M.shape[0]
    v = np.ones(n) if v0 is None else as_vector(v0, "v0").copy()
    if k == 0:
        return v
    if is_sparse(M):
        # Factorized ONCE and reused across the k solves; spsolve would
        # refactorize on every call.
        lu = spla.splu(sparse.csc_array(M) if _is_sparray(M)
                       else sparse.csc_matrix(M))
        for _ in range(k):
            v = lu.solve(v)
        return v
    M = np.asarray(M, dtype=float)
    for _ in range(k):
        v = np.linalg.solve(M, v)
    return v


def stirling_second(n: int, k: int) -> int:
    """
    [NEW] Stirling number of the second kind, in exact integer arithmetic.

        S(n, k) = (1/k!) sum_{j=0}^{k} (-1)^{k-j} C(k, j) j^n
    """
    if n < 0 or k < 0:
        raise ValueError("stirling_second: n and k must be >= 0.")
    if k == 0:
        return 1 if n == 0 else 0
    total = sum((-1) ** (k - j) * comb(k, j) * j ** n for j in range(k + 1))
    return total // factorial(k)


def factorial_to_raw(factorial_moments: Sequence[float]) -> float:
    """
    [NEW] Convert factorial moments into the raw moment of the same order.

        E[X^n] = sum_{k=1}^{n} S(n, k) * E[X(X-1)...(X-k+1)]

    Parameters
    ----------
    factorial_moments : sequence of float
        [E[(X)_1], E[(X)_2], ..., E[(X)_n]] in that order.

    Returns
    -------
    float
        E[X^n], with n = len(factorial_moments).

    Why this is needed
    ------------------
    In the DISCRETE case the closed form yields factorial moments, not raw ones.
    ``AbstractDiscPhaseVar.moment(k)`` in Java returns the factorial moment but
    ``variance()`` uses it as if it were raw, which makes jphase's discrete
    variance incorrect (it is missing the +E[X] term). In the CONTINUOUS case
    this does not apply: there, k! alpha (-A)^{-k} 1 really is the raw moment.
    """
    n = len(factorial_moments)
    if n == 0:
        return 1.0
    return float(
        sum(stirling_second(n, k) * factorial_moments[k - 1] for k in range(1, n + 1))
    )


def format_representation(title: str, alpha, A, matrix_name: str = "A",
                          stats: dict = None) -> str:
    """
    [NEW] Multi-line textual description of a PH representation.

    It is used by ``description()`` in both the continuous and the discrete
    case. In Java each abstract class builds its own string by hand.
    """
    alpha = as_vector(alpha, "alpha")
    n = alpha.shape[0]
    width = 54
    lines = ["_" * width, title, f"Number of phases: {n}", "Vector alpha:"]
    lines.append("\t" + "\t".join(f"{x:8.4f}" for x in alpha))
    a0 = vec0(alpha)
    if abs(a0) > EPS:
        lines.append(f"\talpha_0 (mass at absorption) = {a0:.4f}")
    if is_sparse(A) and n > 12:
        # Not densified just to print it: a summary is shown instead.
        density = A.nnz / (n * n)
        lines.append(f"Matrix {matrix_name}: sparse {type(A).__name__}, "
                     f"nnz={A.nnz:,} ({density:.2%} density)")
    else:
        Ad = to_dense(A)
        lines.append(f"Matrix {matrix_name}:")
        for i in range(n):
            lines.append("\t" + "\t".join(f"{Ad[i, j]:8.4f}" for j in range(n)))
    if stats:
        for key, value in stats.items():
            lines.append(f"{key}: {value:.6f}")
    lines.append("_" * width)
    return "\n".join(lines)
