# Datos — Breast Cancer Wisconsin Diagnostic (WBCD)

El notebook **no incluye un fichero de datos local**: carga el dataset directamente desde
`sklearn.datasets.load_breast_cancer()`, que es una copia **idéntica** (mismas 569 muestras y 30
variables) del dataset original del repositorio UCI. Esto garantiza reproducibilidad y evita
dependencias de red.

## Fuente original (UCI)

- **Nombre:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **URL:** http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Muestras:** 569 (357 benignos · 212 malignos)
- **Variables:** 30 características numéricas + 1 columna `ID` (descartada) + diagnóstico (M/B)
- **Variables:** se calculan 10 magnitudes (radio, textura, perímetro, área, suavidad, compacidad,
  concavidad, puntos cóncavos, simetría, dimensión fractal) en tres agregaciones: *mean*, *error* (se)
  y *worst*.

## Descarga manual desde UCI (opcional)

Si se desea trabajar con el `.csv` original en lugar de la versión de scikit-learn:

```bash
# Descargar y descomprimir el dataset oficial de UCI
curl -L -o wdbc.zip "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
unzip wdbc.zip -d wdbc_raw
# El fichero relevante es wdbc.data (sin cabecera)
```

El fichero `wdbc.data` tiene el formato: `ID, Diagnóstico(M/B), f1, f2, ..., f30`.

## Codificación de la etiqueta usada en el notebook

- **`1 = Maligno (M)`** → clase positiva (interés clínico)
- **`0 = Benigno (B)`** → clase negativa

> ⚠️ `sklearn` usa la convención inversa (`0 = maligno`, `1 = benigno`). El notebook la corrige con
> `y = 1 - y` para que la clase positiva sea el tumor maligno, como exige el análisis clínico.
