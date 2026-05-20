# Proyecto II - Estadística No Paramétrica
**MATE3526 / MATE4532 — 2026-10**

## Descripción

Implementación y evaluación de dos tests bootstrap de simetría para distribuciones univariadas continuas.

### Test 1 — Estadístico tipo Kolmogorov-Smirnov (Schuster & Barker)

Basado en la norma supremo entre la distribución empírica y su simetrización:

$$T_n = \sqrt{n} \| F_n - sF_n(\tilde{\theta}) \|_\infty$$

### Test 2 — Función característica empírica

Basado en una integral de penalización sobre la función característica empírica:

$$S_n = \int_{-\infty}^{\infty} \| c_n(t)e^{-it\hat{\theta}} - c_s^\theta(t) \|^q w(t) \, dt$$

### Estimadores del centro de simetría

Ambos tests consideran tres alternativas para estimar el centro de simetría θ:
- **argmin**: minimizador del estadístico
- **mediana**: `median(X₁, ..., Xₙ)`
- **media afeitada**: `Xα(X₁, ..., Xₙ)`

## Estructura del repositorio

```
├── main.py                        # Código principal de simulación
├── proyecto2_2026.pdf             # Enunciado del proyecto
└── documento/
    ├── main.tex                   # Documento LaTeX principal
    ├── package.sty                # Estilos
    ├── ref.bib                    # Bibliografía
    ├── 0 - Intro.tex
    ├── 1 - Estadisticos.tex
    ├── 2 - metodologia.tex
    ├── 3 - implementacion.tex
    ├── 4 - resultados.tex
    ├── 5 - discusion.tex
    ├── 6 - conclusiones.tex
    ├── 7 - apendice.tex
    └── figs/                      # Figuras del documento
```

## Diseño de experimentos

- **Bajo H₀ (simetría):** Distribución Uniforme (intervalo no centrado en cero) y Cauchy con parámetro de localización distinto de cero.
- **Bajo Hₐ (asimetría, para medir potencia):** Gamma, Weibull o Pareto.
- Se consideran distintos tamaños muestrales `n`.
- Bootstrap con `B = 500` remuestras para obtener p-valores.

## Referencias

1. Arcones, M. A. & Giné, E. (1991). *Some Bootstrap Tests of Symmetry for Univariate Continuous Distributions.* The Annals of Statistics, 19(3), 1496–1511.
2. Csörgő, S. & Heathcote, C. R. (1987). *Testing for symmetry.* Biometrika, 74(1), 177–184.
3. Feuerverger, A. & Mureika, R. A. (1977). *The Empirical Characteristic Function and Its Applications.* The Annals of Statistics, 5(1), 88–97.
4. Schuster, E. F. & Barker, R. C. (1987). *Using the bootstrap in testing symmetry versus asymmetry.* Communications in Statistics - Simulation and Computation, 16(1), 69–84.
