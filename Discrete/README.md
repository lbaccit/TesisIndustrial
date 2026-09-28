# Discrete — implementación Python de distribuciones Phase-Type

Módulo Python para distribuciones Phase-Type discretas, con representaciones
densa y sparse, operaciones de probabilidad, momentos y operaciones de cierre.
Las matrices de prueba continuas se uniformizan para construir los fixtures
discretos; los valores esperados de la suite se generan de forma independiente
con aritmética racional exacta.

## Archivos

| Archivo | Función |
|---|---|
| matrix_utils.py | utilidades matriciales |
| AbstractDiscPhaseVar.py | interfaz y operaciones de una DPH |
| DenseDiscPhaseVar.py | representación densa |
| SparseDiscPhaseVar.py | representación sparse |
| discrete_fixtures.py | fixtures discretos obtenidos por uniformización |
| generate_reference_values.py | referencias independientes en aritmética racional exacta |
| test_*.py | pruebas de teoría, valores y equivalencia entre representaciones |

## Cómo correr los tests

```bash
python3 run_tests.py          # verbose
python3 run_tests.py -q       # una línea por archivo
```

Requiere `numpy` y `scipy`. La suite contiene actualmente 255 tests.

## Discrepancias históricas y decisiones de implementación

Ver [`DISCREPANCIES.md`](DISCREPANCIES.md) — doce discrepancias identificadas
durante la revisión histórica del código de referencia y las decisiones
adoptadas en esta implementación Python. Cada una incluye la fórmula errónea,
la fórmula correcta y la evidencia correspondiente.
