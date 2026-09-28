"""
SparseDiscPhaseVar.py

Discrete Phase-Type distribution with a sparse matrix representation.

Migration of jphase.SparseDiscPhaseVar Java to Python.

Every numerical method (pmf, cdf, moments, quantiles) lives in
AbstractDiscretePhaseType, which already dispatches between dense and sparse
storage through matrix_utils.is_sparse. This class only supplies copy
and new_var, plus the constructor that guarantees the matrix really is
stored sparsely.


Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import numpy as np
import scipy.sparse as sp

# Written twice on purpose: the relative form resolves inside the jmarkov
# package, the absolute form resolves in a flat folder with no __init__.py.
try:
    from .AbstractDiscPhaseVar import AbstractDiscretePhaseType
except ImportError:  # pragma: no cover - depends on the layout, not on logic
    from AbstractDiscPhaseVar import AbstractDiscretePhaseType


class SparseDiscretePhaseType(AbstractDiscretePhaseType):
    """
    DPH(alpha, A) with A stored as a scipy.sparse matrix.

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
    SparseDiscretePhaseType(alpha, dense_A) would silently hold a dense
    matrix and the class name would be misleading.

    Whatever sparse format comes in, the stored matrix ends up as CSR: the
    constructor of the abstract class calls matrix_utils.as_square_matrix,
    which normalizes with .tocsr(). So csc_array in gives csr_array
    stored, and csc_matrix in gives csr_matrix stored; what is preserved
    is the scipy API family (sparray or spmatrix), not the exact format.
    This is the right default here, since every operation on a DPH is row
    oriented (alpha @ A, A @ 1, row sums) and CSR is the format built for that.

    Only the matrix is stored sparsely. The vector alpha is always dense
    """

    def __init__(self, alpha, A):
        if not sp.issparse(A):
            A = sp.csr_array(np.asarray(A, dtype=float))
        super().__init__(alpha, A)

    def copy(self) -> "SparseDiscretePhaseType":
        """
        Independent copy of this variable. Java: copy()

        The arrays are duplicated, so mutating the copy does not affect the
        original. The sparse matrix is copied keeping its format.
        """
        return SparseDiscretePhaseType(self._alpha.copy(), self._A.copy())

    def new_var(self, n: int) -> "SparseDiscretePhaseType":
        """
        Empty variable of the same type with n phases. Java: newVar(int n).

        It returns alpha filled with zeros and A as an all-zero sparse matrix,
        which is a valid if degenerate DPH: with alpha = 0 we get alpha_0 = 1,
        that is P(X = 0) = 1.

        Raises
        ------
        ValueError
            If n < 1.
        """
        
        if n < 1:
            raise ValueError(f"'n' must be >= 1; got {n}.")
        return SparseDiscretePhaseType(np.zeros(n), sp.csr_array((n, n)))
