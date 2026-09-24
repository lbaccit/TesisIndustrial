# Discrete — migración de `jphase` (Java) a Python

Módulo Python que reimplementa la parte **discreta** de `jphase`
(`jMarkov/src/jphase/`): `MatrixUtils.java`, `AbstractDiscPhaseVar.java`,
`DenseDiscPhaseVar.java` y `SparseDiscPhaseVar.java`, junto con su propia
suite de tests (Java no tiene tests para el caso discreto).

## Archivos

| Archivo | Migración de |
|---|---|
| `matrix_utils.py` | `MatrixUtils.java` |
| `AbstractDiscPhaseVar.py` | `AbstractDiscPhaseVar.java` |
| `DenseDiscPhaseVar.py` | `DenseDiscPhaseVar.java` |
| `SparseDiscPhaseVar.py` | `SparseDiscPhaseVar.java` |
| `discrete_fixtures.py` | variables de prueba (uniformización de las matrices de los tests continuos de Java) |
| `generate_reference_values.py` | genera `reference_values.py` en aritmética racional exacta, sin usar el código bajo prueba |
| `test_*.py` | ver tabla de correspondencia en `run_tests.py` |

## Cómo correr los tests

```bash
python3 run_tests.py          # verbose
python3 run_tests.py -q       # una línea por archivo
```

Requiere `numpy` y `scipy`. 268 tests, todos en verde.

## Discrepancias entre la teoría/Java y la implementación

Ver [`DISCREPANCIES.md`](DISCREPANCIES.md) — doce discrepancias encontradas
entre el código Java de jMarkov y la teoría de distribuciones Phase-Type,
cada una con la fórmula errónea, la fórmula correcta, y la evidencia que
prueba el error (derivación matemática, contraejemplo numérico reproducible,
o comparación con el código análogo continuo del mismo repositorio).
