# Semana 1 — Clasificación de Cáncer de Mama (WBCD): Regresión Logística, KNN y Naive Bayes

**Práctica grupal — Semana 1**
**Materia:** Aprendizaje Automático Estadístico — MIA-B
**Docente:** PhD. Iván García Santillán
**Maestría:** Inteligencia Artificial Aplicada (MIA-B) — UIDE

---

## Objetivo

Construir un flujo completo de **clasificación binaria supervisada** sobre el dataset *Breast Cancer
Wisconsin Diagnostic* (WBCD) para distinguir tumores **malignos** de **benignos**, comparando cuatro
algoritmos — **Regresión Logística**, **KNN**, **Naive Bayes** y **Random Forest** (este último a
propuesta del equipo) — en versión *baseline* y *mejorada* (optimización con `GridSearchCV` +
selección de características).

> **Criterio clínico:** la clase positiva (1) es el tumor **maligno**. La métrica prioritaria es el
> **Recall (sensibilidad)**, porque el error más grave es el **Falso Negativo** (no detectar un cáncer).

## Dataset

- **Fuente:** [Breast Cancer Wisconsin Diagnostic — UCI](http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (vía `sklearn.datasets.load_breast_cancer()`)
- **Muestras:** 569 (357 benignos / 212 malignos)
- **Variables:** 30 características morfológicas numéricas
- **Valores perdidos:** 0 · **Atípicos IQR:** 171 filas (30,1%, conservados)

## Estructura

```
semana-1/
├── notebook/
│   └── S1_Cancer_Mama_Clasificacion.ipynb   # Notebook completo y ejecutado
├── models/
│   ├── logistic_regression_model.pkl        # Modelos optimizados (joblib)
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   └── random_forest_model.pkl
├── data/
│   └── README.md                            # Instrucciones de descarga (UCI)
├── informe/
│   └── S1_Informe_Interpretacion.md         # Informe ejecutivo
├── outputs/                                  # Figuras generadas (.png) + CSV de resultados
└── README.md
```

## Cómo ejecutar

```bash
# Desde la raíz del repositorio
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd semana-1/notebook
jupyter notebook S1_Cancer_Mama_Clasificacion.ipynb
# o ejecutar de forma headless:
jupyter nbconvert --to notebook --execute --inplace S1_Cancer_Mama_Clasificacion.ipynb
```

Los modelos entrenados se guardan en `models/` y las figuras en `outputs/`.

## Resultados

| Modelo | Versión | Accuracy | Precision | Recall | F1 | AUC |
|--------|---------|:--------:|:---------:|:------:|:--:|:---:|
| Reg. Logística | Baseline | 0,9649 | 0,9750 | 0,9286 | 0,9512 | 0,9960 |
| **Reg. Logística** | **Mejorado** | **0,9825** | **0,9762** | **0,9762** | **0,9762** | 0,9954 |
| KNN | Baseline | 0,9561 | 0,9744 | 0,9048 | 0,9383 | 0,9823 |
| KNN | Mejorado | 0,9649 | 1,0000 | 0,9048 | 0,9500 | 0,9861 |
| Naive Bayes | Baseline | 0,9211 | 0,9231 | 0,8571 | 0,8889 | 0,9891 |
| Naive Bayes | Mejorado | 0,9386 | 0,9730 | 0,8571 | 0,9114 | 0,9921 |
| Random Forest | Baseline | 0,9737 | 1,0000 | 0,9286 | 0,9630 | 0,9929 |
| Random Forest | Mejorado | 0,9649 | 1,0000 | 0,9048 | 0,9500 | 0,9937 |

**Mejor modelo: Regresión Logística mejorada** (`C=10, solver='saga'`) — mejor Recall y F1 (0,976),
es decir, la que comete **menos Falsos Negativos**. **Random Forest** (propuesta del equipo) queda
como mejor alternativa, con un hallazgo instructivo: su *baseline* (Recall 0,929) superó a su versión
optimizada (0,905) en el test, recordando que el óptimo de validación cruzada no siempre transfiere.

## Análisis adicionales (más allá de lo solicitado)

- Comparación empírica de normalizadores `StandardScaler` vs `MinMaxScaler`.
- Experimento "eliminar vs conservar atípicos" con justificación clínica.
- Cuarto algoritmo (**Random Forest**) con doble lectura de importancia: coeficientes (lineal) vs Gini.
- Análisis crítico del caso *baseline > mejorado* en Random Forest (sobre-optimización en CV).
- Lectura clínica de los errores (FN/FP) de la matriz de confusión.

📓 [Ver notebook](notebook/S1_Cancer_Mama_Clasificacion.ipynb) · 📄 [Informe ejecutivo](informe/S1_Informe_Interpretacion.md)
