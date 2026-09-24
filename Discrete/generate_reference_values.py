"""
generate_reference_values.py

Writes ``reference_values.py``, the expected values used by the discrete test
suite. It does not import the package under test: every value is computed
here, independently, in exact rational arithmetic (``fractions.Fraction``) and
only rounded to float when written out.

Where each group of values comes from
-------------------------------------
1. pmf / cdf / quantiles (Prob tests). The absorbing chain is advanced one
   step at a time: pi_0 = alpha, pi_k = pi_{k-1} A, and the mass absorbed at
   step k is pi_{k-1} a. This is the definition of the DPH, not the closed
   forms alpha A^{k-1} a and 1 - alpha A^k 1 that the code uses.
2. Moments (Moment tests). Factorial moments k! alpha (I-A)^{-k} A^{k-1} 1
   solved exactly by Gauss-Jordan elimination over the rationals, converted
   to raw moments with Stirling numbers of the second kind, and cross-checked
   against the series sum_k k^j P(X = k).
3. Closure representations (Closure tests). The textbook constructions,
   written out again here with exact arithmetic, the same way
   DenseContClosureTest hard-codes the matrices it expects.
4. jMarkov's own reference values (Java gold). The realPDF/realCDF tables and
   the moments hard-coded in DenseContProbTest.java and DenseContMomentTest.java,
   copied verbatim. The discrete fixtures are uniformizations of those
   continuous variables, so the tests can check them against these tables
   (see test_DenseDiscProb.py / test_DenseDiscMoment.py).

Usage
-----
    python3 generate_reference_values.py [PATH_TO_jMarkov/test/jphase]

Without a path, the Java tables already in ``reference_values.py`` are kept.

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import os
import re
import sys
from fractions import Fraction
from math import comb, factorial

from discrete_fixtures import MIX_P, SUM_GEOM_P, fixture

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reference_values.py")

PROB_K_MAX = {"prob1": 40, "prob2": 40, "prob3": 60}
QUANTILE_PS = ["0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"]

# Java grids: DenseContProbTest evaluates var_i at t = index * step.
JAVA_PROB = {"prob1": ("testProb1", "0.1"), "prob2": ("testProb2", "0.1"),
             "prob3": ("testProb3", "0.25")}
# prob3's table has 4001 entries (t up to 1000); every 40th one is kept.
JAVA_PROB_STRIDE = {"prob1": 1, "prob2": 1, "prob3": 40}


# -----------------------------------------------------------------------------
# Exact linear algebra on lists of Fractions
# -----------------------------------------------------------------------------

def zeros(r, c):
    return [[Fraction(0)] * c for _ in range(r)]


def eye(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def vecmat(v, A):
    return [sum(v[k] * A[k][j] for k in range(len(v))) for j in range(len(A[0]))]


def matvec(A, v):
    return [sum(A[i][k] * v[k] for k in range(len(v))) for i in range(len(A))]


def dot(u, v):
    return sum(x * y for x, y in zip(u, v))


def add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def scale(c, A):
    return [[c * a for a in row] for row in A]


def outer(u, v):
    return [[x * y for y in v] for x in u]


def kron(A, B):
    return [[A[i][j] * B[k][l] for j in range(len(A[0])) for l in range(len(B[0]))]
            for i in range(len(A)) for k in range(len(B))]


def kron_vec(u, v):
    return [x * y for x in u for y in v]


def block(rows):
    out = []
    for row in rows:
        for r in range(len(row[0])):
            out.append([x for blk in row for x in blk[r]])
    return out


def solve(M, b):
    """Solve M x = b exactly (Gauss-Jordan with pivoting on nonzeros)."""
    n = len(M)
    aug = [list(M[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def inverse(M):
    n = len(M)
    cols = [solve(M, [Fraction(int(i == j)) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ones(n):
    return [Fraction(1)] * n


def mat0(A):
    return [1 - sum(row) for row in A]


def vec0(alpha):
    return 1 - sum(alpha)


# -----------------------------------------------------------------------------
# 1. pmf / cdf / quantiles via the absorbing chain
# -----------------------------------------------------------------------------

def chain_pmf(alpha, A, k_max):
    a = mat0(A)
    pmf = [vec0(alpha)]
    pi = list(alpha)
    for _ in range(k_max):
        pmf.append(dot(pi, a))
        pi = vecmat(pi, A)
    return pmf


def quantiles(alpha, A, ps):
    out = {}
    for p_str in ps:
        p = Fraction(p_str)
        acc, k, pi, a = vec0(alpha), 0, list(alpha), mat0(A)
        while acc < p:
            k += 1
            acc += dot(pi, a)
            pi = vecmat(pi, A)
        out[p_str] = k
    return out


# -----------------------------------------------------------------------------
# 2. Moments
# -----------------------------------------------------------------------------

def stirling2(n, k):
    return sum((-1) ** (k - j) * comb(k, j) * j ** n for j in range(k + 1)) // factorial(k)


def factorial_moments(alpha, A, k_max):
    n = len(A)
    I_minus_A = add(eye(n), scale(Fraction(-1), A))
    res = []
    v = ones(n)                       # A^{k-1} 1
    for k in range(1, k_max + 1):
        w = v
        for _ in range(k):
            w = solve(I_minus_A, w)   # (I-A)^{-k} A^{k-1} 1
        res.append(factorial(k) * dot(alpha, w))
        v = matvec(A, v)
    return res


def raw_moments(fm):
    return [sum(stirling2(n, k) * fm[k - 1] for k in range(1, n + 1))
            for n in range(1, len(fm) + 1)]


def check_moments_against_series(alpha, A, raw):
    """Float cross-check: sum_k k^j P(X=k), far enough for the tail to vanish."""
    import numpy as np
    al = np.array([float(x) for x in alpha])
    Af = np.array([[float(x) for x in row] for row in A])
    a = 1 - Af.sum(axis=1)
    v, series = al.copy(), [0.0] * len(raw)
    for k in range(1, 20000):
        pk = v @ a
        for j in range(len(raw)):
            series[j] += k ** (j + 1) * pk
        v = v @ Af
    for j, exact in enumerate(raw):
        assert abs(series[j] - float(exact)) <= 1e-9 * float(exact), (j, series[j], exact)


# -----------------------------------------------------------------------------
# 3. Closure representations (textbook constructions)
# -----------------------------------------------------------------------------

def closure_sum(X, Y):
    (al, A), (be, B) = X, Y
    n1, n2 = len(A), len(B)
    alpha = al + [x * vec0(al) for x in be]
    M = block([[A, outer(mat0(A), be)], [zeros(n2, n1), B]])
    return alpha, M


def closure_min(X, Y):
    (al, A), (be, B) = X, Y
    return kron_vec(al, be), kron(A, B)


def closure_max(X, Y):
    (al, A), (be, B) = X, Y
    n1, n2 = len(A), len(B)
    alpha = kron_vec(al, be) + [x * vec0(be) for x in al] + [x * vec0(al) for x in be]
    a, b = mat0(A), mat0(B)
    to_x_only = kron(A, [[x] for x in b])      # A (x) b, shape (n1 n2, n1)
    to_y_only = kron([[x] for x in a], B)      # a (x) B, shape (n1 n2, n2)
    M = block([
        [kron(A, B), to_x_only, to_y_only],
        [zeros(n1, n1 * n2), A, zeros(n1, n2)],
        [zeros(n2, n1 * n2), zeros(n2, n1), B],
    ])
    return alpha, M


def closure_mix(p, X, Y):
    (al, A), (be, B) = X, Y
    n1, n2 = len(A), len(B)
    alpha = [p * x for x in al] + [(1 - p) * x for x in be]
    return alpha, block([[A, zeros(n1, n2)], [zeros(n2, n1), B]])


def closure_sum_geom(p, X):
    al, A = X
    c = 1 / (1 - (1 - p) * vec0(al))
    return [c * x for x in al], add(A, scale((1 - p) * c, outer(mat0(A), al)))


def closure_sum_ph(X, N):
    (al, A), (be, S) = X, N
    n2 = len(S)
    M = inverse(add(eye(n2), scale(-vec0(al), S)))
    Mt_beta = [sum(M[k][j] * be[k] for k in range(n2)) for j in range(n2)]
    return (kron_vec(al, Mt_beta),
            add(kron(A, eye(n2)), kron(outer(mat0(A), al), matmul(M, S))))


# -----------------------------------------------------------------------------
# 4. Java gold tables
# -----------------------------------------------------------------------------

NUMBER = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"


def java_prob_tables(java_dir):
    src = open(os.path.join(java_dir, "DenseContProbTest.java"),
               encoding="latin-1").read()
    out = {}
    for name, (method, step) in JAVA_PROB.items():
        body = src.split(f"public void {method}()")[1]
        pdf = re.search(r"realPDF\s*=\s*\{(.*?)\};", body, re.S).group(1)
        cdf = re.search(r"realCDF\s*=\s*\{(.*?)\};", body, re.S).group(1)
        pdf = [float(x) for x in re.findall(NUMBER, pdf)]
        cdf = [float(x) for x in re.findall(NUMBER, cdf)]
        stride = JAVA_PROB_STRIDE[name]
        idx = list(range(0, min(len(pdf), len(cdf)), stride))
        out[name] = {"step": step, "index": idx,
                     "pdf": [pdf[i] for i in idx], "cdf": [cdf[i] for i in idx]}
    return out


def java_moments(java_dir):
    src = open(os.path.join(java_dir, "DenseContMomentTest.java"),
               encoding="latin-1").read()
    grab = lambda name: float(re.search(rf"{name}\s*=\s*({NUMBER});", src).group(1))
    return {"mean": grab("realEV"), "var": grab("realVar"), "cv": grab("realCV"),
            "m2": grab("realM2"), "m3": grab("realM3"), "m4": grab("realM4")}


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def f(x):
    return repr(float(x))


def fmt_vec(v, indent=8, per_line=4):
    items = [f(x) for x in v]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    pad = " " * indent
    return "[\n" + ",\n".join(pad + ln for ln in lines) + ",\n" + " " * (indent - 4) + "]"


def fmt_mat(M, indent=8):
    pad = " " * indent
    rows = [pad + "[" + ", ".join(f(x) for x in row) + "]," for row in M]
    return "[\n" + "\n".join(rows) + "\n" + " " * (indent - 4) + "]"


def main():
    java_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if java_dir:
        java_tables, java_mom = java_prob_tables(java_dir), java_moments(java_dir)
    else:
        import reference_values as old
        java_tables, java_mom = old.JAVA_CONT_PROB, old.JAVA_CONT_MOMENTS

    out = ['"""',
           "reference_values.py",
           "",
           "GENERATED by generate_reference_values.py - do not edit by hand.",
           "Regenerate with:  python3 generate_reference_values.py [jMarkov/test/jphase]",
           "",
           "Every value except JAVA_CONT_* was computed in exact rational arithmetic",
           "and rounded to float only when written. JAVA_CONT_* are copied verbatim",
           "from jMarkov's DenseContProbTest.java and DenseContMomentTest.java.",
           '"""', ""]

    out.append("PROB = {")
    for name, k_max in PROB_K_MAX.items():
        alpha, A = fixture(name, exact=True)
        pmf = chain_pmf(alpha, A, k_max)
        cdf, acc = [], Fraction(0)
        for p in pmf:
            acc += p
            cdf.append(acc)
        q = quantiles(alpha, A, QUANTILE_PS)
        out.append(f"    {name!r}: {{")
        out.append(f"        'pmf': {fmt_vec(pmf, 12)},")
        out.append(f"        'cdf': {fmt_vec(cdf, 12)},")
        out.append(f"        'quantile': {{{', '.join(f'{p}: {k}' for p, k in q.items())}}},")
        out.append("    },")
    out.append("}")
    out.append("")

    alpha, A = fixture("moment", exact=True)
    fm = factorial_moments(alpha, A, 4)
    raw = raw_moments(fm)
    check_moments_against_series(alpha, A, raw)
    mean, m2 = raw[0], raw[1]
    var = m2 - mean * mean
    std = float(var) ** 0.5
    out.append("MOMENT = {")
    out.append(f"    'mean': {f(mean)},")
    out.append(f"    'var': {f(var)},")
    out.append(f"    'std': {std!r},")
    out.append(f"    'cv': {std / float(mean)!r},")
    out.append(f"    'scv': {f(var / (mean * mean))},")
    out.append(f"    'raw': {{{', '.join(f'{k + 1}: {f(x)}' for k, x in enumerate(raw))}}},")
    out.append(f"    'factorial': {{{', '.join(f'{k + 1}: {f(x)}' for k, x in enumerate(fm))}}},")
    out.append("}")
    out.append("")

    X1 = fixture("closure1", exact=True)
    X2 = fixture("closure2", exact=True)
    XD = fixture("closureD", exact=True)
    p_mix, p_geom = Fraction(str(MIX_P)), Fraction(str(SUM_GEOM_P))
    closures = {
        "sum": closure_sum(X1, X2),
        "min": closure_min(X1, X2),
        "max": closure_max(X1, X2),
        "mix": closure_mix(p_mix, X1, X2),
        "sum_geom": closure_sum_geom(p_geom, X2),
        "sum_ph": closure_sum_ph(X2, XD),
    }
    out.append("CLOSURE = {")
    for name, (al, M) in closures.items():
        out.append(f"    {name!r}: {{")
        out.append(f"        'alpha': {fmt_vec(al, 12)},")
        out.append(f"        'A': {fmt_mat(M, 12)},")
        out.append("    },")
    out.append("}")
    out.append("")

    out.append("JAVA_CONT_PROB = {")
    for name, t in java_tables.items():
        out.append(f"    {name!r}: {{")
        out.append(f"        'step': {t['step']!r},")
        out.append(f"        'index': {t['index']!r},")
        out.append(f"        'pdf': {fmt_vec(t['pdf'], 12)},")
        out.append(f"        'cdf': {fmt_vec(t['cdf'], 12)},")
        out.append("    },")
    out.append("}")
    out.append("")
    out.append(f"JAVA_CONT_MOMENTS = {java_mom!r}")
    out.append("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
