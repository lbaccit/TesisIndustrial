"""
SparseDistPhaseVar.py

Discrete Phase-Type distribution with a sparse matrix representation.

Migration of ``jphase.SparseDiscPhaseVar`` (Java) to Python.

Every numerical method (pmf, cdf, moments, quantiles) lives in
``AbstractDiscretePhaseType``, which already dispatches between dense and sparse
storage through ``matrix_utils.is_sparse``. This class only supplies ``copy``
and ``new_var``, plus the constructor that guarantees the matrix really is
stored sparsely.

When sparse storage pays off
----------------------------
A DPH sub-stochastic matrix is often mostly zeros: an Erlang or a hyper-Erlang
is bidiagonal, a hyperexponential is diagonal. For a bidiagonal sub-generator
with n = 10,000, dense storage takes about 800 MB against roughly 0.3 MB in CSR.
Below a few hundred phases the dense version is usually faster, because sparse
formats carry indexing overhead; the gain shows up on large, genuinely sparse
matrices.

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import numpy as np
import scipy.sparse as sp

# Written twice on purpose: the relative form resolves inside the jmarkov
# package, the absolute form resolves in a flat folder with no __init__.py.
try:
    from .AbstractDistPhaseVar import AbstractDiscretePhaseType
except ImportError:  # pragma: no cover - depends on the layout, not on logic
    from AbstractDistPhaseVar import AbstractDiscretePhaseType


class SparseDiscretePhaseType(AbstractDiscretePhaseType):
    """
    DPH(alpha, A) with A stored as a ``scipy.sparse`` matrix.

    Parameters
    ----------
    alpha : array_like
        1-D vector of initial probabilities over the transient phases. It must
        satisfy alpha >= 0 and sum(alpha) <= 1. Vectors are always stored dense
        (see the note below).
    A : array_like or scipy.sparse matrix
        Sub-stochastic n x n matrix. It must satisfy A >= 0, row sums <= 1 and
        sp(A) < 1. Any accepted input ends up stored as CSR (see the note on
        formats below).

    Notes
    -----
    Dense input is converted rather than stored as given. Otherwise
    ``SparseDiscretePhaseType(alpha, dense_A)`` would silently hold a dense
    matrix and the class name would be misleading. Java does not need this
    because the caller there builds a ``FlexCompRowMatrix`` explicitly, so the
    type itself already carries the decision.

    Whatever sparse format comes in, the stored matrix ends up as CSR: the
    constructor of the abstract class calls ``matrix_utils.as_square_matrix``,
    which normalizes with ``.tocsr()``. So ``csc_array`` in gives ``csr_array``
    stored, and ``csc_matrix`` in gives ``csr_matrix`` stored; what is preserved
    is the scipy API family (``sparray`` or ``spmatrix``), not the exact format.
    This is the right default here, since every operation on a DPH is row
    oriented (alpha @ A, A @ 1, row sums) and CSR is the format built for that.

    Only the matrix is stored sparsely. The vector alpha is always dense: it
    costs O(n) against the O(n^2) of the matrix, so the saving would be marginal
    while complicating every downstream operation.

    Examples
    --------
    Geometric with success probability 0.5, stored sparsely:

    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> X = SparseDiscretePhaseType(np.array([1.0]), sp.csr_array([[0.5]]))
    >>> round(X.mean(), 6), round(X.var(), 6)
    (2.0, 2.0)
    >>> [round(X.pmf(k), 4) for k in range(4)]
    [0.0, 0.5, 0.25, 0.125]

    Negative binomial as two phases in series:

    >>> alpha = np.array([1.0, 0.0])
    >>> A = sp.csr_array([[0.6, 0.4], [0.0, 0.6]])
    >>> Y = SparseDiscretePhaseType(alpha, A)
    >>> round(Y.mean(), 6), round(Y.var(), 6)
    (5.0, 7.5)

    A dense input is accepted and converted:

    >>> Z = SparseDiscretePhaseType(np.array([1.0]), np.array([[0.5]]))
    >>> sp.issparse(Z.A)
    True

    Any sparse format is normalized to CSR, keeping the scipy API family:

    >>> W = SparseDiscretePhaseType(np.array([1.0]), sp.csc_array([[0.5]]))
    >>> type(W.A).__name__
    'csr_array'
    """

    def __init__(self, alpha, A):
        if not sp.issparse(A):
            A = sp.csr_array(np.asarray(A, dtype=float))
        super().__init__(alpha, A)

    def copy(self) -> "SparseDiscretePhaseType":
        """
        Independent copy of this variable. Java: ``copy()``.

        The arrays are duplicated, so mutating the copy does not affect the
        original. The sparse matrix is copied keeping its format.
        """
        return SparseDiscretePhaseType(self._alpha.copy(), self._A.copy())

    def new_var(self, n: int) -> "SparseDiscretePhaseType":
        """
        Empty variable of the same type with `n` phases. Java: ``newVar(int n)``.

        It returns alpha filled with zeros and A as an all-zero sparse matrix,
        which is a valid if degenerate DPH: with alpha = 0 we get alpha_0 = 1,
        that is P(X = 0) = 1.

        The closure operations (sum, mixture, minimum, maximum) use it to create
        the container for the result without knowing the concrete subclass, and
        then fill it with the corresponding blocks.

        Raises
        ------
        ValueError
            If n < 1.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> X = SparseDiscretePhaseType(np.array([1.0]), sp.csr_array([[0.5]]))
        >>> Z = X.new_var(3)
        >>> Z.n_phases, Z.A.nnz, round(Z.get_vec0(), 6)
        (3, 0, 1.0)
        """
        if n < 1:
            raise ValueError(f"'n' must be >= 1; got {n}.")
        return SparseDiscretePhaseType(np.zeros(n), sp.csr_array((n, n)))
