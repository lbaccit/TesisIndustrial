"""
AbstractDistPhaseVar.py

Abstract class for discrete Phase-Type distributions.

Migration of jphase.AbstractDiscPhaseVar from Java to Python.

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

from abc import ABC, abstractmethod
from math import factorial, sqrt

import numpy as np

try:
    from .matrix_utils import (
        EPS,
        check_sub_stochastic_matrix,
        check_sub_stochastic_vector,
        coerce_representation,
        factorial_to_raw,
        format_representation,
        eye_like,
        is_sparse,
        mat0,
        mat_power,
        ones_vector,
        solve_power,
        vec0,
    )
except ImportError:
    from matrix_utils import (
        EPS,
        check_sub_stochastic_matrix,
        check_sub_stochastic_vector,
        coerce_representation,
        factorial_to_raw,
        format_representation,
        eye_like,
        is_sparse,
        mat0,
        mat_power,
        ones_vector,
        solve_power,
        vec0,
    )


class AbstractDiscretePhaseType(ABC):
    """
    Discrete Phase-Type distribution DPH(alpha, A).

    Parameters
    ----------
    alpha : array_like
        1-D vector of initial probabilities over the transient phases. It must
        satisfy alpha >= 0 and sum(alpha) <= 1.
    A : array_like
        Sub-stochastic n x n matrix of transitions between transient phases. It
        must satisfy A >= 0, row sums <= 1, and sp(A) < 1.

    Attributes
    ----------
    n_phases : int
        Number of transient phases.

    """

    def __init__(self, alpha, A):
        self._alpha, self._A, self.n_phases = coerce_representation(alpha, A)

        if not check_sub_stochastic_vector(self._alpha):
            raise ValueError(
                "'alpha' is not a valid defective probability vector: "
                "alpha_i >= 0 and sum(alpha) <= 1 are required. "
                f"Got sum(alpha) = {self._alpha.sum():.6g}."
            )

        if not check_sub_stochastic_matrix(self._A):
            raise ValueError(
                "'A' is not a valid transient sub-stochastic matrix. "
                "Required: A_ij >= 0, row sums <= 1, and sp(A) < 1 "
                "(this last one guarantees that absorption happens with "
                f"probability 1). sp(A) = "
                f"{np.max(np.abs(np.linalg.eigvals(self._A))):.6g}."
            )



    @property
    def alpha(self) -> np.ndarray:
        """Vector of initial probabilities."""
        return self._alpha.copy()

    @property
    def A(self) -> np.ndarray:
        """Sub-stochastic matrix."""
        return self._A.copy()

    def get_mat0(self) -> np.ndarray:
        """
        Absorption vector a = 1 - A*1.

        a_i is the probability of being absorbed in one step from phase i.
        copy from Java: ``getMat0()``.
        """
        return mat0(self._A, discrete=True)

    def get_vec0(self) -> float:
        """
        P(X = 0) = alpha_0 = 1 - alpha*1.

        copy from Java: ``getVec0()``.
        """
        return vec0(self._alpha)


    def pmf(self, k: int) -> float:
        """
        Probability mass function.

            P(X = 0) = alpha_0
            P(X = k) = alpha A^{k-1} a, k >= 1

        Parameters
        ----------
        k : int
            Evaluation point, k >= 0.

        """
        k = self._check_index(k)
        if k == 0:
            return self.get_vec0()
        return mat_power(self._A, k - 1, self._alpha, self.get_mat0())

    def pmf_range(self, k_max: int) -> np.ndarray:
        """
        PMF evaluated at k = 0, 1, ..., k_max.

        Parameters
        ----------
        k_max : int
            Maximum value of k to evaluate.

        Returns
        -------
        np.ndarray
            Array of length k_max + 1.
        """
        k_max = self._check_index(k_max, name="k_max")
        a = self.get_mat0()
        res = np.zeros(k_max + 1)
        res[0] = self.get_vec0()
        v = self._alpha.copy()          # v = alpha A^{k-1}
        for k in range(1, k_max + 1):
            res[k] = v @ a
            v = v @ self._A
        return res

    def cdf(self, k: int) -> float:
        """
        Cumulative distribution function.

            F(k) = P(X <= k) = 1 - alpha A^k 1

        For k = 0 it reduces to alpha_0, consistent with ``pmf(0)``.
        """
        k = self._check_index(k)
        return 1.0 - mat_power(self._A, k, self._alpha, ones_vector(self.n_phases))

    def cdf_range(self, k_max: int) -> np.ndarray:
        """
        CDF evaluated at k = 0, 1, ..., k_max (incremental propagation).
        """
        return np.cumsum(self.pmf_range(k_max))

    def survival(self, k: int) -> float:
        """
        Survival function.

            S(k) = P(X > k) = alpha A^k 1
        """
        k = self._check_index(k)
        return mat_power(self._A, k, self._alpha, ones_vector(self.n_phases))

    def prob(self, a: int, b: int) -> float:
        """
        P(a < X <= b) = F(b) - F(a). Returns 0.0 if b <= a.
        """
        if b <= a:
            return 0.0
        return self.cdf(b) - self.cdf(a)

    # -------------------------------------------------------------------
    # Moments
    # -------------------------------------------------------------------

    def factorial_moment(self, k: int) -> float:
        """
        k-th descending factorial moment.

            E[X(X-1)...(X-k+1)] = k! alpha (I-A)^{-k} A^{k-1} 1

        This is the moment that comes out in closed form for a DPH; the raw
        moments E[X^k] are derived from these (see ``moment``).

        Important
        ---------
        This method reproduces exactly what ``AbstractDiscPhaseVar.moment(k)``
        computes in Java. See the note in ``moment`` about the difference.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}.")
        # A^{k-1} 1, then (I - A)^{-k} applied k times. (I-A)^{-1} and A commute
        # (both are polynomials in A), so the order is free.
        v = mat_power(self._A, k - 1) @ ones_vector(self.n_phases)
        I_minus_A = eye_like(self._A) - self._A
        v = solve_power(I_minus_A, k, v)
        return float(factorial(k) * (self._alpha @ v))

    def moment(self, k: int = 1) -> float:
        """
        k-th raw moment E[X^k].

        It is obtained from the factorial moments through Stirling numbers of
        the second kind:

            E[X^k] = sum_{j=1}^{k} S(k, j) E[X(X-1)...(X-j+1)]

        Warning about the comparison against Java
        -----------------------------------------
        ``AbstractDiscPhaseVar.moment(k)`` in Java returns the FACTORIAL moment,
        not the raw one, even though ``variance()`` and ``CV()`` there use it as
        if it were raw. For k = 1 they coincide; for k >= 2 they do not. Example
        with Geometric(0.5) - alpha = [1], A = [[0.5]]:

            E[X]        = 2      (Java and Python agree)
            E[X(X-1)]   = 4      (Java moment(2) = our factorial_moment(2))
            E[X^2]      = 6      (our moment(2))
            True Var    = 2
            Java var()  = 4 - 2^2 = 0      <-- incorrect

        If you need to reproduce Java bit for bit during validation, use
        ``factorial_moment(k)``. This point is worth raising with the advisor.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}.")
        fmoments = [self.factorial_moment(j) for j in range(1, k + 1)]
        return factorial_to_raw(fmoments)

    def mean(self) -> float:
        """
        Expected value E[X] = alpha (I-A)^{-1} 1.

        It coincides with ``moment(1)`` and with ``factorial_moment(1)``.
        """
        I_minus_A = eye_like(self._A) - self._A
        v = solve_power(I_minus_A, 1)
        return float(self._alpha @ v)

    def var(self) -> float:
        """
        Variance Var(X) = E[X^2] - E[X]^2.

        Equivalently, in terms of factorial moments:
        Var(X) = E[X(X-1)] + E[X] - E[X]^2.
        """
        m1 = self.mean()
        return self.factorial_moment(2) + m1 - m1 * m1

    def std(self) -> float:
        """Standard deviation."""
        return sqrt(self.var())

    def cv(self) -> float:
        """
        Coefficient of variation CV = sd(X) / E[X].
        """
        return self.std() / self.mean()

    def scv(self) -> float:
        """
        Squared coefficient of variation, SCV = Var(X) / E[X]^2.

        Note: ``AbstractDiscPhaseVar.CV()`` in Java computes moment(2)/m^2 - 1,
        which is the form of the SCV (not of the CV) - and on top of that with
        the factorial moment instead of the raw one.
        """
        m = self.mean()
        return self.var() / (m * m)

    def quantile(self, p: float, k_max: int = 10 ** 6) -> int:
        """
        Quantile of order p: the smallest k with F(k) >= p.

        Java: ``quantil(double p)``.

        Parameters
        ----------
        p : float
            Order of the quantile, in (0, 1].
        k_max : int
            Search cap. It is only reached if p is extremely close to 1 or if
            sp(A) is almost 1.

        Returns
        -------
        int

        Divergence from Java
        --------------------
        ``quantil`` in Java applies Newton-Raphson using ``pmf`` as if it were
        the derivative of ``cdf``, that is, it treats a step function as if it
        were differentiable. Worse: if it does not converge within 100
        iterations it returns ``0.0`` instead of reporting a failure. With a
        Geometric(0.5) that happens for p = 0.9, 0.95 and 0.99, where the
        correct answers are 4, 5 and 7.

        For a discrete variable the quantile follows from the definition by
        accumulating the pmf, which is also exact. That is what is done here.
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"'p' must be in (0, 1]; got {p}.")
        accumulated = self.get_vec0()
        if accumulated >= p - EPS:
            return 0
        a = self.get_mat0()
        v = self._alpha.copy()
        for k in range(1, k_max + 1):
            accumulated += float(v @ a)
            if accumulated >= p - EPS:
                return k
            v = v @ self._A
        raise RuntimeError(
            f"quantile: p={p} was not reached within {k_max} steps. Raise "
            "'k_max' or check whether sp(A) is too close to 1."
        )

    def median(self) -> int:
        """Median, that is the quantile of order 0.5. Java: ``median()``."""
        return self.quantile(0.5)

    # -------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------

    def label(self) -> str:
        """Short label."""
        return f"DPH - {self.n_phases} phases"

    def description(self) -> str:
        """Full description of the representation and its statistics."""
        return format_representation(
            title="Discrete Phase-Type distribution",
            alpha=self._alpha,
            A=self._A,
            matrix_name="A",
            stats={"Mean": self.mean(), "Variance": self.var(), "CV": self.cv()},
        )

    def __str__(self) -> str:
        return self.label()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_phases={self.n_phases})"

    def __eq__(self, other) -> bool:
        """
        Equality of representations (not of distributions).

        Two different representations can generate the same distribution; this
        method compares the parameters, not the probability law.
        """
        if not isinstance(other, AbstractDiscretePhaseType):
            return NotImplemented
        if self.n_phases != other.n_phases:
            return False
        if not np.allclose(self._alpha, other._alpha, atol=EPS):
            return False
        if is_sparse(self._A) or is_sparse(other._A):
            # Sparse structures are not compared entry by entry: two
            # representations with a different format or a different pattern of
            # explicit zeros can still be the same matrix. What is compared is
            # the difference, which stays sparse.
            diff = self._A - other._A
            largest = abs(diff).max() if getattr(diff, "nnz", 1) else 0.0
            return bool(largest <= EPS)
        return bool(np.allclose(self._A, other._A, atol=EPS))

    def __hash__(self):
        A = self._A
        fingerprint = (A.data.tobytes(), A.indices.tobytes()) if is_sparse(A) \
            else (A.tobytes(),)
        return hash((type(self).__name__, self.n_phases,
                     self._alpha.tobytes()) + fingerprint)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _check_index(k, name: str = "k") -> int:
        """Validate that `k` is a non-negative integer."""
        if isinstance(k, (bool, np.bool_)):
            raise TypeError(f"'{name}' must be an integer, not a bool.")
        if isinstance(k, (float, np.floating)):
            if abs(k - round(k)) > EPS:
                raise ValueError(
                    f"'{name}' must be an integer; got {k}. The DPH is only "
                    "defined over the non-negative integers."
                )
            k = int(round(k))
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"'{name}' must be an integer; got {type(k).__name__}.")
        if k < 0:
            raise ValueError(f"'{name}' must be >= 0; got {k}.")
        return int(k)

    # -------------------------------------------------------------------
    # Abstract methods
    # -------------------------------------------------------------------

    @abstractmethod
    def copy(self) -> "AbstractDiscretePhaseType":
        """Return an independent copy of this variable."""

    @abstractmethod
    def new_var(self, n: int) -> "AbstractDiscretePhaseType":
        """
        Build an empty variable of the same concrete type with n phases.

        It is used by the closure operations (sum, mixture, minimum, maximum)
        to create the container for the result without knowing the subclass.
        """
