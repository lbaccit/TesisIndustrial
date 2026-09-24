# Discrepancias entre la teoría, jMarkov (Java) y esta implementación (Python)

Este documento lista los errores encontrados en `jphase` (Java) durante la
migración de la parte discreta a Python, para cada uno: **ubicación exacta**,
**fórmula/código erróneo**, **fórmula/código correcto**, y la **evidencia**
que prueba el error.

## Metodología (qué cuenta como "fuente" aquí)

Para cada discrepancia se da uno o más de estos tres tipos de evidencia,
**todos verificables de forma independiente sin confiar en este texto**:

1. **Prueba matemática directa** — se deriva el error a partir de la
   definición de la operación (álgebra lineal, análisis dimensional), sin
   necesidad de ejecutar nada. Es la evidencia más fuerte: es una
   demostración, no una opinión.
2. **Verificación numérica independiente** — un contraejemplo concreto,
   calculado por dos caminos que no comparten código: la fórmula bajo
   sospecha vs. una definición de fuerza bruta (ej. convolución directa,
   acumulación paso a paso de la cadena de Markov). Los números de esta
   tabla se generaron con los scripts en `Discrete/` y son reproducibles.
3. **Comparación con el código análogo** — cuando `jphase` tiene una versión
   continua (`AbstractContPhaseVar.java`) y una discreta
   (`AbstractDiscPhaseVar.java`) del mismo método, y ambas son
   *textualmente* idénticas, es evidencia fuerte de que el código discreto
   fue copiado sin adaptar (no es una decisión de diseño).

Cuando se cita una referencia teórica general (Neuts 1981; Latouche &
Ramaswami 1999) es para señalar dónde vive la construcción estándar
correspondiente, **no como cita verificada de página**: no tuve acceso al
texto completo de esos libros en esta sesión. La prueba real de cada error
es (1)-(3), no la referencia bibliográfica.

## Convenciones

- `alpha`, `A`: vector inicial y matriz sub-estocástica de la distribución
  Phase-Type discreta DPH(alpha, A).
- `alpha_0 = 1 - alpha·1`: masa en cero, `P(X=0)`.
- `a = mat0(A) = 1 - A·1`: vector de absorción (prob. de absorber en un paso
  desde cada fase).
- Todos los contraejemplos numéricos se pueden reproducir así:
  ```python
  import numpy as np
  from DenseDiscPhaseVar import DenseDiscretePhaseType as D
  X = D(np.array([...]), np.array([[...]]))
  ```

---

## A. Errores en `MatrixUtils.java`

### A.1 — `kroneckerMxRowVector`: stride de columna incorrecto

- **Ubicación Java:** `MatrixUtils.java:187-224`
- **Ubicación Python (corregido):** `matrix_utils.kronecker_mx_row_vector`
- **Estado:** código muerto — nada en `jphase` lo llama.

**Código erróneo (Java):**
```java
res.set(i1*b1+i2, j1*c1+i2, A.get(i1,j1)*B.get(i2));
```
escribe en la columna `j1*c1 + i2`, donde `c1` es el número de **columnas**
de `A`.

**Código correcto:**
la definición de producto de Kronecker requiere la columna `j1*n2 + i2`,
donde `n2` es el tamaño de `B` (no `c1`).

**Evidencia — prueba matemática directa:** las dos fórmulas solo coinciden
cuando `A` es cuadrada con `c1 == n2`. Contraejemplo: `A` de 2×3, `b` de
tamaño 2. Java escribe en las columnas `{0,1,3,4,6,7}` de una matriz 2×6:
dos columnas (`2` y `5`) quedan sin llenar y dos índices (`6`, `7`) exceden
el ancho de la matriz para alguna fila.

---

### A.2 — `sumMatPower`: no acumula las potencias

- **Ubicación Java:** `MatrixUtils.java:1392-1408`
- **Ubicación Python (corregido):** `matrix_utils.sum_mat_power`
- **Estado:** código muerto — nada en `jphase` lo llama.

**Código erróneo (Java):**
```java
Matrix temp = Matrices.identity(n);
Matrix sum  = temp.copy();
for (int i = 1; i < k; i++) {
    temp.mult(A, temp.copy());   // resultado descartado
    sum.add(temp);
}
```

**Prueba matemática directa:** en MTJ, `X.mult(B, C)` calcula `C = X·B` y
**devuelve `C`**, pero **no modifica `X`**. Como el resultado de
`temp.mult(A, temp.copy())` nunca se reasigna a `temp`, `temp` permanece
igual a la identidad en cada iteración. Por lo tanto `sum` termina siendo
`k·I`, no `I + A + A² + ... + A^(k-1)`.

**Verificación numérica:** `A = [[0.5]]`, `k = 3`.

| | Java (código tal cual) | Correcto (`sum_mat_power`) |
|---|---|---|
| Resultado | `3·I = [[3.0]]` | `I+A+A² = [[1.75]]` |

**Código correcto (Python):**
```python
total = eye_like(A); term = eye_like(A)
for _ in range(1, k):
    term = term @ A
    total = total + term
```

---

### A.3 — `checkSubGeneratorMatrix`: tres huecos de validación

- **Ubicación Java:** `MatrixUtils.java:1435-1457`
- **Ubicación Python (corregido):** `matrix_utils.check_sub_generator_matrix`
- **Estado:** se usa (validación de matrices generadoras continuas), pero de forma laxa.

**Código Java:**
```java
public static boolean checkSubGeneratorMatrix(Matrix A){
    boolean res = true;
    double n = A.numRows(), m = A.numColumns();
    if (n == m){
        for (int i = 0; i < n; i++){
            double cumsum = A.get(i,0);
            for (int j = 1; j < m; j++){          // <- empieza en j=1
                cumsum += A.get(i,j);
                if ((i==j && A.get(i,j) > -Epsilon) || (i!=j && A.get(i,j) < -Epsilon)){
                    res = false; i += n; break;
                }
            }
            if (cumsum > Epsilon){ res = false; break; }
        }
    }
    return res;
}
```

Tres problemas, cada uno con contraejemplo:

**(a) La columna 0 nunca se inspecciona** (`j` arranca en 1): ni
`A[0][0] < 0` ni `A[i][0] ≥ 0` se verifican para `j=0`.
*Contraejemplo:* `A = [[5, -5], [0, -0.01]]` (columna 0 con signo
invertido) — Java lo acepta porque el chequeo de signo nunca toca `j=0`, y
la suma de la fila 0 es `0 ≤ Epsilon`.

**(b) No exige salida a absorción.** La condición 4 de una matriz
generadora válida (al menos una fila con suma `< 0`) nunca se verifica.
*Contraejemplo:* un generador completo (todas las filas suman exactamente
0, sin ninguna vía de absorción) pasa la validación aunque el vector de
absorción `a = -A·1` sea idénticamente cero y la variable nunca termine.

**(c) No rechaza matrices no cuadradas.** `res` parte en `true` y el bloque
de validación solo corre `if (n == m)`; si `n ≠ m`, la función devuelve
`true` sin haber revisado nada.
*Contraejemplo:* `A` de 1×3 pasa `checkSubGeneratorMatrix` trivialmente.

**Fuente teórica general:** las cuatro condiciones de una matriz
sub-generadora válida (no-negatividad fuera de diagonal, diagonal negativa,
filas suman ≤ 0, al menos una fila suma < 0) son estándar en la teoría de
cadenas de Markov de tiempo continuo con absorción — Neuts (1981);
Latouche & Ramaswami (1999).

---

### A.4 — `checkSubStochasticVector`: nunca revisa el signo de `a[0]`

- **Ubicación Java:** `MatrixUtils.java:1415-1428`
- **Ubicación Python (corregido):** `matrix_utils.check_sub_stochastic_vector`
- **Estado:** se usa (validación de `alpha`).

**Código erróneo (Java):**
```java
double cumsum = a.get(0);
for (int i = 1; i < a.size(); i++){   // <- empieza en i=1
    if (a.get(i) < -Epsilon){ nonneg = false; break; }
    cumsum += a.get(i);
}
```
`a[0]` se suma a `cumsum` pero jamás se compara contra `-Epsilon`.

**Verificación numérica:** `a = [-5, 0.5]`.

| | Java | Correcto |
|---|---|---|
| `cumsum` | `-4.5 ≤ 1+ε` → `true` | se rechaza por `a[0] = -5 < 0` |
| Resultado | **acepta** un vector con una entrada negativa | rechaza |

---

### A.5 — `CV(double[])`: en realidad calcula el SCV, no el CV

- **Ubicación Java:** `MatrixUtils.java:1073-1077`
- **Ubicación Python:** `matrix_utils.cv` (nombre y comportamiento
  conservados por trazabilidad) y `matrix_utils.cv_true` (el CV real,
  agregado).

**Código Java:**
```java
public static double CV(double[] data) {
    double m = average(data);
    return variance(data) / (m*m);
}
```

**Prueba matemática directa:** `Var/mean²` es el **cuadrado** del
coeficiente de variación (SCV), no el CV. El CV es `sqrt(Var)/mean`. No es
un bug numérico (el cálculo es internamente consistente), es un error de
nomenclatura: cualquier código que llame `CV()` esperando `sd/mean` recibe
`Var/mean²` en su lugar — para valores típicos con `Var/mean² ≠ 1`, la
diferencia es sustancial (ej. `Var/mean²=4` da `CV=2` correctamente, pero
`sqrt(4)=2` mientras que el valor devuelto es `4`).

---

## B. Errores en `AbstractDiscPhaseVar.java`

### B.1 — `cdf(x)`: usa el vector de absorción en vez del vector de unos

- **Ubicación Java:** `AbstractDiscPhaseVar.java:166-170`
- **Ubicación Python (corregido):** `AbstractDiscPhaseVar.cdf`

**Código erróneo (Java):**
```java
public double cdf(double x) {
    int k = (int) x;
    if (k==0) return this.getVec0();
    return 1 - MatrixUtils.matPower(this.getMatrix(), k, this.getVector(), this.getMat0());
}
```
usa `getMat0()` (`a`, vector de absorción) donde debería usar el vector de
unos `1`.

**Prueba matemática directa:** `alpha·Aᵏ·a` es, por definición,
`P(X = k+1)` (es `pmf(k+1)`, ver `pmf(k) = alpha·A^(k-1)·a`), **no**
`P(X > k) = alpha·Aᵏ·1` (la supervivencia). La fórmula correcta de la CDF
es `F(k) = 1 - alpha·Aᵏ·1`.

**Verificación numérica:** Geométrica(0.5) — `alpha=[1]`, `A=[[0.5]]`.

| | Valor |
|---|---|
| `pmf(1)` | `0.5` |
| `F(1)` verdadero (`P(X≤1) = alpha_0 + pmf(1)`) | `0 + 0.5 = 0.5` |
| Fórmula de Java: `1 - alpha·A¹·mat0` | `1 - 0.5·0.5 = 0.75` ❌ |
| `cdf()` (Python, corregido) | `0.5` ✓ |

Como `survival(x) = 1 - cdf(x)` en Java se define en términos de `cdf`,
hereda el mismo error.

**Fuente teórica general:** `S(k) = P(X>k) = alpha·Aᵏ·1` es la definición
estándar de función de supervivencia para una DPH — Neuts (1981);
Latouche & Ramaswami (1999).

---

### B.2 — `moment(k)` devuelve el momento factorial, pero `variance()`/`CV()` lo usan como momento crudo

- **Ubicación Java:** `AbstractDiscPhaseVar.java:120-138` (`variance`, `CV`) y `:143-160` (`moment`)
- **Ubicación Python (corregido):** `AbstractDiscPhaseVar.moment` (momento crudo) / `AbstractDiscPhaseVar.factorial_moment` (momento factorial, reproduce a Java)

**Código Java:**
```java
public double variance() {
    double m1 = moment(1);
    return moment(2) - (m1*m1);
}
public double moment(int k) {
    // ... calcula k! * alpha * (I-A)^-k * A^(k-1) * 1
}
```
`moment(k)` calcula el **momento factorial descendente**
`E[X(X-1)...(X-k+1)]`, no el momento crudo `E[X^k]`. Para `k=1` ambos
coinciden, pero no para `k≥2`.

**Verificación numérica:** Geométrica(0.5) — `alpha=[1]`, `A=[[0.5]]`.

| | Valor |
|---|---|
| `E[X]` | `2.0` |
| `E[X(X-1)]` = `moment(2)` de Java | `4.0` |
| `E[X²]` verdadero (momento crudo) | `6.0` |
| `Var(X)` verdadero | `2.0` |
| `variance()` de Java = `moment(2) - E[X]²` | `4.0 - 4.0 = 0.0` ❌ |

Una varianza de `0` para una variable no degenerada es, por sí sola,
matemáticamente imposible — no hace falta ninguna referencia externa para
ver que el resultado está mal.

**Corrección (Python):** los momentos crudos se recuperan de los momentos
factoriales con los números de Stirling de segunda especie:
`E[X^n] = Σ_{k=1}^{n} S(n,k)·E[X(X-1)...(X-k+1)]` — identidad combinatoria
estándar (`matrix_utils.factorial_to_raw`).

---

### B.3 — `quantil(p)`: Newton-Raphson sobre una función escalón

- **Ubicación Java:** `AbstractDiscPhaseVar.java:254-267`
- **Ubicación Python (corregido):** `AbstractDiscPhaseVar.quantile`

**Código Java:**
```java
public double quantil(double p) {
    double x = this.expectedValue();
    for (int cnt = 0; cnt < 100; cnt++) {
        double derv = this.pmf((int)x);
        double dif = this.cdf(x) - p;
        if (Math.abs(dif) < 1.0e-10) return x;
        x = Math.max(x - dif/derv, 0);
    }
    return 0.0;   // <- si no converge, devuelve 0 en silencio
}
```

**Prueba matemática directa:** `cdf(x)` es una función escalón (constante a
tramos, con saltos en los enteros); no es diferenciable, por lo que
Newton-Raphson (que asume que `pmf` es la derivada de una `cdf` suave) no
tiene garantía de converger. Además, si no converge en 100 iteraciones, el
método devuelve `0.0` **sin lanzar ninguna excepción ni aviso**,
indistinguible de un cuantil legítimo igual a 0.

**Verificación numérica:** Geométrica(0.5).

| `p` | Cuantil verdadero (`min k : F(k)≥p`) | `quantil()` de Java |
|---|---|---|
| 0.9 | 4 | `0.0` ❌ (no converge) |
| 0.95 | 5 | `0.0` ❌ |
| 0.99 | 7 | `0.0` ❌ |

---

### B.4 — `sumPH`: invierte una matriz n2×n2 resolviendo contra la identidad n1×n1

- **Ubicación Java:** `AbstractDiscPhaseVar.java:25-51`, específicamente la línea 39
- **Ubicación Python (corregido):** `AbstractDiscPhaseVar.sum_ph`

**Código erróneo (Java):**
```java
Matrix ISInv = Matrices.identity(n2).add(-this.getVec0(), B.getMatrix());  // n2×n2
ISInv = ISInv.solve(Matrices.identity(this.getNumPhases()), ISInv.copy()); // identidad de tamaño n1
```
`this.getNumPhases()` es `n1` (fases de `this`), no `n2` (fases de `B`,
el contador).

**Prueba matemática directa:** resolver `M·X = C` (`Matrix.solve(C, X)` en
MTJ) exige que `C` tenga tantas filas como `M`. Como `ISInv` (el `M` en esta
llamada) es de tamaño `n2×n2`, la llamada solo tiene sentido dimensional
cuando `n1 == n2`; para cualquier otro caso el número de filas no
coincide.

**Comparación con el código análogo:** la versión continua del mismo
método, `AbstractContPhaseVar.sumPH` (línea 24-54), calcula exactamente lo
mismo pero con `ISInv.solve(Matrices.identity(n2), ISInv.copy())` — usa
`n2` correctamente. Es el mismo autor, la misma fórmula, la misma
estructura de código; solo la versión discreta tiene el error, lo que
indica una transcripción defectuosa entre ambas (no una decisión de
diseño).

**Verificación funcional (Python):** con `n1=2` fases para `X` y `n2=2`
fases para el contador `N` (`n1 ≠ n2`, el caso que en Java no correría),
`X.sum_ph(N)` produce una variable de `n1·n2 = 4` fases, verificado contra
la definición por fuerza bruta (ver B.6 más abajo).

---

### B.5 — `max()`: copiado del caso continuo sin adaptar a tiempo discreto

- **Ubicación Java:** `AbstractDiscPhaseVar.java:413-463` vs. `AbstractContPhaseVar.java:472-522`
- **Ubicación Python (corregido):** `AbstractDiscPhaseVar.max`

**Comparación con el código análogo:** las dos implementaciones son
**textualmente idénticas** (mismas variables, mismo orden de operaciones,
mismos nombres): bloque conjunto construido con `kroneckerSum(A, B)` y
transiciones de frontera con `kronecker(I_n1, mat0(B))` /
`kronecker(mat0(A), I_n2)`.

**Prueba matemática directa (por qué es correcto en continuo pero no en
discreto):**

- **Tiempo continuo:** con probabilidad 1, como mucho un subproceso
  transiciona en un instante dado, así que las tasas de salida se suman —
  de ahí la suma de Kronecker — y cuando un subproceso absorbe, la fase del
  otro *no cambia* durante ese mismo evento instantáneo — de ahí la
  identidad junto a cada `mat0`.
- **Tiempo discreto:** en cada paso, **ambos** subprocesos transicionan
  simultáneamente, y **ambos pueden absorber en el mismo paso**. El bloque
  "ambos siguen corriendo" debe ser literalmente el **producto** de
  Kronecker `A⊗B` (igual que hace correctamente `min()`, que no fue
  copiado del caso continuo), y en las transiciones de frontera la
  variable que sobrevive también avanza en ese mismo paso, así que debe
  usarse su propia matriz, no la identidad.

**Verificación numérica:** `X = Geométrica(0.5)`, `Y = Geométrica(0.7)`
(ambas de 1 fase, `A=[[0.5]]`, `B=[[0.3]]`, `mat0(A)=[0.5]`, `mat0(B)=[0.7]`).

| | Fórmula de Java | Fórmula correcta |
|---|---|---|
| Bloque conjunto (`kroneckerSum` vs. `kronecker`) | `0.5+0.3 = 0.8` | `0.5·0.3 = 0.15` |
| Suma de fila completa (bloque conjunto + fronteras) | **`0.8+0.7+0.5 = 2.0`** ❌ | `0.15+0.5·0.7+0.5·0.3 = 0.65` ✓ |
| Masa faltante en la fila (debe ser `mat0(A)·mat0(B)`) | `1-2.0 = -1.0` (¡negativa!) | `1-0.65 = 0.35 = 0.5·0.7` ✓ |

Una fila de una matriz de transición cuya suma es `2.0` es, por definición,
imposible en cualquier cadena de Markov — esto prueba el error sin
necesidad de referencia externa.

**Fuente teórica general:** la construcción de Kronecker para el mínimo y
el máximo de variables Phase-Type independientes es estándar — Neuts
(1981); Latouche & Ramaswami (1999) — pero la distinción tiempo-continuo
(suma de Kronecker) vs. tiempo-discreto (producto de Kronecker) es la que
falta en el código de Java.

---

### B.6 — `sumGeom(p)`: falta el factor de normalización cuando `alpha_0 > 0`

- **Ubicación Java:** `AbstractDiscPhaseVar.java:332-341` (idéntico a `AbstractContPhaseVar.java:384-392`)
- **Ubicación Python (corregido):** `AbstractDiscPhaseVar.sum_geom`

**Código erróneo (Java):**
```java
public DiscPhaseVar sumGeom(double p) {
    DiscPhaseVar res = this.copy();
    res.setVector(this.getVector());                     // alpha_res = alpha
    res.setMatrix(this.getMatrix().copy().add(1-p,
        MatrixUtils.multVector(this.getMat0(), this.getVector(), ...)));  // A + (1-p)·a⊗alpha
    return res;
}
```

**Prueba matemática directa:** cada vez que una copia de `X` absorbe, con
probabilidad `(1-p)` arranca otra copia. Pero una copia nueva puede *ella
misma* nacer ya absorbida (con probabilidad `alpha_0`), en cuyo caso el
proceso vuelve a tirar la moneda `(1-p)` de inmediato — una cadena
potencialmente infinita de copias de longitud cero. La fórmula de Java
ignora esas cadenas por completo. El factor que las suma es
`c = 1/(1-(1-p)·alpha_0) = Σ_{m≥0} ((1-p)·alpha_0)^m`, y falta tanto en
`alpha_res` como en el término de reinicio de `A_res`:

```
alpha_res = c · alpha
A_res     = A + (1-p)·c · (a ⊗ alpha)
```

Cuando `alpha_0 = 0` (la variable nunca nace ya absorbida) `c = 1` y ambas
fórmulas coinciden — por eso el error nunca se manifestó: no hay ningún
test, ni en Java ni originalmente en Python, que use una variable con masa
en cero para `sumGeom`.

**Verificación numérica:** `X` de 1 fase, `alpha=[0.5]` (`alpha_0=0.5`),
`A=[[0.3]]`, `p=0.5`. Se compara contra la definición por fuerza bruta:
suma de una cantidad Geométrica(p) de copias de `X`, calculada por
convolución directa de la pmf (sin usar ninguna fórmula cerrada).

| | `alpha_res` | `A_res` | `P(Y=0)` | `E[Y]` (Wald: `E[X]/p`) |
|---|---|---|---|---|
| Fuerza bruta (verdad) | — | — | `0.3333` | `1.4286` |
| Fórmula de Java | `0.5` | `0.475` | `0.5` ❌ | `0.9524` ❌ |
| Fórmula correcta | `0.6667` | `0.5333` | `0.3333` ✓ | `1.4286` ✓ |

Incluso `P(Y=0)` — el número más simple de la distribución — difiere en
`0.5` vs. `0.333`, un error de 50%.

**Nota:** el método `sumGeom` de `AbstractContPhaseVar.java` (continuo)
tiene el mismo código, línea por línea — ver arriba, código Java citado en
la sección "Comparación con el código análogo" de B.5 no aplica aquí
directamente, pero el archivo `AbstractContPhaseVar.java:384-392` reproduce
la misma fórmula sin el factor `c`. El propio test de Java
`DenseContClosureTest.testSumGeom` usa `var2` con `vector2={0.1,0.2,0.2}`
(suma `0.5`, es decir `alpha_0=0.5 > 0`) — el caso exacto donde el error se
manifiesta. El valor `matRes` que ese test da como "correcto" fue obtenido
ejecutando el código de Java, así que reproduce el error por construcción;
no se tomó como referencia para nada en este puerto.

---

### B.7 — `sumPH`: factor `(1-alpha_0)` de más, duplicado

- **Ubicación Java:** `AbstractDiscPhaseVar.java:43-48` (la llamada `.scale(1-this.getVec0())`)
- **Ubicación Python (corregido):** `AbstractDiscPhaseVar.sum_ph`

**Código erróneo (Java):**
```java
Matrix L2 = MatrixUtils.kronecker(
    (MatrixUtils.multVector(this.getMat0(), this.getVector(), ...)).scale(1 - this.getVec0()),
    ISInv.mult(B.getMatrix(), ...),
    res.getMatrix());
```

**Prueba matemática directa:** el término de reinicio debería ser
`(a ⊗ alpha) ⊗ (M·S)`, sin ningún factor extra. `getVector()` ya es el
vector `alpha` **incondicional** (incluye la masa `alpha_0` de nacer
absorbida); multiplicarlo además por `(1-alpha_0)` cuenta esa probabilidad
dos veces. La fórmula sería correcta si `getVector()` devolviera el vector
condicional `alpha/(1-alpha_0)` (la distribución de fase *dado que* la
copia no nace absorbida), pero no es lo que devuelve.

**Verificación numérica:** `X` de 1 fase, `alpha=[0.5]` (`alpha_0=0.5`),
`A=[[0.3]]`; contador `N` de 2 fases, `beta=[0.6,0.4]`,
`S=[[0.1,0.2],[0,0.3]]`. Comparado contra la definición por fuerza bruta
(convolución ponderada por la pmf exacta del contador).

| | `E[Y]` (verdad: `E[X]·E[N]`) | máx. diferencia en pmf vs. fuerza bruta |
|---|---|---|
| Fórmula de Java (con el factor `1-alpha_0` de más) | `0.9217` ❌ | `0.0363` |
| Fórmula correcta (sin ese factor) | `1.0204` ✓ | `< 10⁻¹⁵` |
| Verdad (`E[X]·E[N] = 0.5/0.7 · 1.4286`) | `1.0204` | — |

---

## Resumen

| # | Método | Archivo Java | Expuesto por (test discreto, Python) |
|---|---|---|---|
| A.1 | `kroneckerMxRowVector` | `MatrixUtils.java:187` | sin test de regresión dedicado (código muerto); documentado en el docstring de `kronecker_mx_row_vector` |
| A.2 | `sumMatPower` | `MatrixUtils.java:1392` | `test_matrix_utils.TestSumMatPower.test_sum_mat_power_java_bug` |
| A.3 | `checkSubGeneratorMatrix` | `MatrixUtils.java:1435` | `test_matrix_utils.TestValidations` |
| A.4 | `checkSubStochasticVector` | `MatrixUtils.java:1415` | `test_matrix_utils.TestValidations.test_check_sub_stochastic_vector_checks_first_entry` |
| A.5 | `CV(double[])` | `MatrixUtils.java:1073` | `test_matrix_utils.TestDistanceScalarStats.test_cv_is_scv_like_java` |
| B.1 | `cdf(x)` | `AbstractDiscPhaseVar.java:166` | `test_AbstractDiscPhaseVar.TestPmfCdfConsistency` |
| B.2 | `moment`/`variance`/`CV` | `AbstractDiscPhaseVar.java:120-160` | `test_AbstractDiscPhaseVar.TestGeometricAnalytic` |
| B.3 | `quantil(p)` | `AbstractDiscPhaseVar.java:254` | `test_AbstractDiscPhaseVar.TestGeometricAnalytic.test_quantile_does_not_reproduce_java_bug` |
| B.4 | `sumPH` (dimensión) | `AbstractDiscPhaseVar.java:39` | `test_DenseDiscClosure.DenseDiscClosureJavaDivergenceTest.test_sum_ph_works_when_phase_counts_differ` |
| B.5 | `max()` | `AbstractDiscPhaseVar.java:413` | `test_DenseDiscClosure.DenseDiscClosureJavaDivergenceTest.test_max_java_formula_is_not_substochastic` |
| B.6 | `sumGeom(p)` | `AbstractDiscPhaseVar.java:332` | `test_DenseDiscClosure.DenseDiscClosureJavaDivergenceTest.test_sum_geom_java_formula_fails_with_mass_at_zero` |
| B.7 | `sumPH` (factor extra) | `AbstractDiscPhaseVar.java:43-48` | `test_DenseDiscClosure.DenseDiscClosureJavaDivergenceTest.test_sum_ph_java_formula_fails_with_mass_at_zero` |

Todos los contraejemplos numéricos de este documento son reproducibles
ejecutando los scripts usados para generarlos (disponibles en el historial
de esta sesión) o los tests de regresión listados en la última columna.

## Alcance de esta revisión

Este documento cubre lo que se tocó al migrar el caso **discreto**
(`MatrixUtils.java`, `AbstractDiscPhaseVar.java`, `Dense/SparseDiscPhaseVar.java`).
No se revisó el resto de `jphase` (fitting, generadores, GUI) ni se verificó
exhaustivamente el caso continuo — donde se menciona (B.5, B.6) es porque la
comparación con el código continuo fue evidencia directa del error discreto,
no porque se haya auditado el archivo continuo completo.
