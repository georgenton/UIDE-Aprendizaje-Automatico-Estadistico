# Semana 1 — Clasificación supervisada: Cáncer de Mama (WBCD) y Titanic

**Práctica grupal — Semana 1**
**Materia:** Aprendizaje Automático Estadístico — MIA-B
**Docente:** PhD. Iván García Santillán
**Maestría:** Inteligencia Artificial Aplicada (MIA-B) — UIDE

---

## Objetivo

La entrega comprende **dos notebooks** de clasificación binaria supervisada, cada uno comparando cuatro
algoritmos — **Regresión Logística**, **KNN**, **Naive Bayes** y **Random Forest** — en versión
*baseline* y *mejorada* (`GridSearchCV` + selección de características):

1. **Cáncer de Mama (WBCD):** distinguir tumores **malignos** de **benignos** (dataset clínico curado).
2. **Titanic:** predecir la **supervivencia** de los pasajeros (dataset "sucio": nulos, categóricas,
   *feature engineering*), que aporta el contraste de un problema más difícil y menos separable.

Ambos comparten el mismo formato, funciones de evaluación y criterios de preprocesamiento.

> **Criterio de métrica:** en cáncer la prioridad es el **Recall** (minimizar Falsos Negativos, no
> dejar pasar un maligno); en Titanic, donde los costos de error son más simétricos, se prioriza el
> **F1**.

## Datasets

- **WBCD:** [Breast Cancer Wisconsin Diagnostic — UCI](http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (vía `sklearn.datasets.load_breast_cancer()`). 569 muestras (357 benignos / 212 malignos), 30 variables, 0 nulos.
- **Titanic:** dataset clásico de Kaggle (`titanic_train.csv` en `data/`). 891 pasajeros (549 no sobrevivieron / 342 sí), 12 columnas originales con nulos en `Age`, `Cabin` y `Embarked`.

## Estructura

```
semana-1/
├── notebook/
│   ├── S1_Cancer_Mama_Clasificacion.ipynb   # Notebook WBCD (ejecutado)
│   └── S1_Titanic_Clasificacion.ipynb        # Notebook Titanic (ejecutado)
├── models/                                    # 8 modelos optimizados (joblib): 4 WBCD + 4 titanic_*
├── data/
│   ├── README.md                             # Instrucciones de descarga (UCI)
│   └── titanic_train.csv                     # Dataset Titanic
├── informe/
│   └── S1_Informe_Interpretacion.md          # Informe ejecutivo (WBCD)
├── outputs/                                   # Figuras (.png) + CSV de resultados (WBCD y t* Titanic)
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

## Resultados — Cáncer de Mama (WBCD)

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

## Resultados — Titanic

Dataset más difícil (señal más débil): las métricas son más bajas y la **prioridad es F1**.

| Modelo | Versión | Accuracy | Precision | Recall | F1 | AUC |
|--------|---------|:--------:|:---------:|:------:|:--:|:---:|
| Reg. Logística | Baseline | 0,8045 | 0,7833 | 0,6812 | 0,7287 | 0,8513 |
| Reg. Logística | Mejorado | 0,7765 | 0,7164 | 0,6957 | 0,7059 | 0,8171 |
| KNN | Baseline | 0,8156 | 0,7903 | 0,7101 | 0,7481 | 0,8498 |
| KNN | Mejorado | 0,7709 | 0,7333 | 0,6377 | 0,6822 | 0,7924 |
| Naive Bayes | Baseline | 0,7821 | 0,7143 | 0,7246 | 0,7194 | 0,8194 |
| Naive Bayes | Mejorado | 0,7598 | 0,6857 | 0,6957 | 0,6906 | 0,7904 |
| **Random Forest** | **Baseline** | 0,8101 | 0,7612 | 0,7391 | **0,7500** | 0,8291 |
| Random Forest | Mejorado | 0,7821 | 0,7206 | 0,7101 | 0,7153 | 0,8198 |

**Mejor modelo: Random Forest baseline** (F1 = 0,750), seguido muy de cerca por **KNN baseline**
(F1 = 0,748; mejor Accuracy y AUC) — prácticamente empatados. Hallazgo consistente con el de cáncer:
**ninguna versión mejorada superó a su baseline** en el test, porque con pocas variables y señal débil
el baseline ya opera cerca del techo del problema.

## Análisis adicionales (más allá de lo solicitado)

- Comparación empírica de normalizadores `StandardScaler` vs `MinMaxScaler`.
- Experimento "eliminar vs conservar atípicos" con justificación clínica.
- Cuarto algoritmo (**Random Forest**) con doble lectura de importancia: coeficientes (lineal) vs Gini.
- Análisis crítico del caso *baseline > mejorado* en Random Forest (sobre-optimización en CV).
- Lectura clínica de los errores (FN/FP) de la matriz de confusión.
- Segundo dataset (**Titanic**) con *feature engineering* (`FamilySize`, `IsAlone`) y contraste
  fácil vs difícil entre ambos problemas.

📓 [Notebook WBCD](notebook/S1_Cancer_Mama_Clasificacion.ipynb) · 📓 [Notebook Titanic](notebook/S1_Titanic_Clasificacion.ipynb) · 📄 [Informe ejecutivo](informe/S1_Informe_Interpretacion.md) · 🧩 [Notas de integración](NOTAS_INTEGRACION.md)
