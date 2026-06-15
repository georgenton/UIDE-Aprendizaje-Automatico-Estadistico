# Informe ejecutivo — Clasificación de Cáncer de Mama (WBCD)

**Maestría en Inteligencia Artificial Aplicada — UIDE**
**Asignatura:** Aprendizaje Automático Estadístico — MIA-B · **Semana 1**
**Docente:** PhD. Iván García Santillán
**Grupo 5 · Autor principal:** Jorge Quizamánchuro

---

## 1. Resumen del dataset y preprocesamiento aplicado

Se trabajó con el **Breast Cancer Wisconsin Diagnostic (WBCD)**: 569 muestras, 30 características
morfológicas numéricas calculadas sobre imágenes de aspirados con aguja fina (FNA) y un diagnóstico
binario. La etiqueta se recodificó para fijar la **clase positiva en el tumor maligno**
(`1 = maligno`, `0 = benigno`), invirtiendo la convención de scikit-learn (`y = 1 - y`).

Preprocesamiento aplicado y verificado:

- **Columna ID:** se confirmó su ausencia en la versión de scikit-learn (en UCI sería descartada por no
  aportar poder predictivo).
- **Valores perdidos:** **0** en todo el dataset → no se requirió imputación.
- **Atípicos (IQR):** **171 de 569 filas (30,1%)** contienen al menos un valor atípico, concentrados en
  variables de tipo *error* (`area error`, `radius error`) y en la clase maligna.
- **Normalización:** se probaron `StandardScaler` y `MinMaxScaler`; se adoptó **StandardScaler** como
  preprocesamiento único por su idoneidad para modelos lineales y de distancia (KNN).

## 2. Hallazgos del EDA

- **Balance de clases:** desbalance **moderado** (357 benignos / 212 malignos ≈ 63% / 37%). No exige
  remuestreo, pero obliga a reportar **recall y F1** de la clase maligna además del accuracy.
- **Variables más predictivas:** las relacionadas con **tamaño** y **forma/concavidad** dominan. Las
  cinco de mayor correlación con malignidad son `worst concave points` (r = +0,79), `worst perimeter`
  (+0,78), `mean concave points` (+0,78), `worst radius` (+0,78) y `mean perimeter` (+0,74).
- **Selección de características:** se retuvieron las **20 de 30** variables con |r| > 0,4 frente al
  objetivo, descartando las menos informativas (errores de textura, simetría y dimensión fractal).
- **Multicolinealidad alta** entre las variables de tamaño (radio/perímetro/área correlacionan > 0,9):
  justifica la selección de características y advierte cautela al interpretar coeficientes.
- **Separabilidad:** los histogramas muestran clases bien separadas → se anticipa (y se confirma) un
  desempeño alto incluso para modelos lineales simples.

## 3. Comparativa de los tres algoritmos

| Modelo | Versión | Accuracy | Precision | Recall | F1 | AUC |
|--------|---------|:--------:|:---------:|:------:|:--:|:---:|
| Reg. Logística | Baseline | 0,9649 | 0,9750 | 0,9286 | 0,9512 | 0,9960 |
| **Reg. Logística** | **Mejorado** | **0,9825** | **0,9762** | **0,9762** | **0,9762** | 0,9954 |
| KNN | Baseline | 0,9561 | 0,9744 | 0,9048 | 0,9383 | 0,9823 |
| KNN | Mejorado | 0,9649 | 1,0000 | 0,9048 | 0,9500 | 0,9861 |
| Naive Bayes | Baseline | 0,9211 | 0,9231 | 0,8571 | 0,8889 | 0,9891 |
| Naive Bayes | Mejorado | 0,9386 | 0,9730 | 0,8571 | 0,9114 | 0,9921 |

**Ganador: Regresión Logística mejorada** (`C=10, solver='saga', max_iter=500`), con el mejor
**Recall (0,976)** y el mejor **F1 (0,976)** — es decir, la que **menos Falsos Negativos** comete, que
es exactamente el criterio prioritario en oncología.

**Razonamiento clínico y estadístico:**
- La **Regresión Logística** gana porque el problema es linealmente muy separable y la frontera lineal
  generaliza bien; además su recall mejorado significa que solo deja escapar 1 de cada 42 malignos del
  conjunto de prueba.
- **KNN** alcanza precisión perfecta (1,0) tras la optimización (`k=3, manhattan`), pero su recall se
  queda en 0,905: es más "conservador" detectando malignos, lo que clínicamente es **peor** (más FN).
- **Naive Bayes** es el más débil: su suposición de **independencia entre variables** choca con la
  fuerte multicolinealidad del WBCD, penalizando el recall (0,857). Buen contraste didáctico.

## 4. Decisiones de preprocesamiento y su justificación

1. **Clase positiva = maligno:** alinea las métricas con el objetivo clínico (detectar cáncer).
2. **Conservar los atípicos:** el experimento mostró que eliminarlos mejora *marginalmente*
   accuracy/F1 en esta partición pero **con recall idéntico**, a costa de descartar ~30% del
   entrenamiento y la señal de los tumores más agresivos. Se conservan por robustez y prudencia
   estadística (evitar sobreajuste a una sola división).
3. **StandardScaler:** recall idéntico al de MinMaxScaler; se elige por idoneidad para KNN/lineales y
   por mantener un preprocesamiento único y coherente para los tres modelos.
4. **Selección por correlación (|r|>0,4):** reduce dimensionalidad y multicolinealidad; especialmente
   beneficioso para KNN (maldición de la dimensionalidad).
5. **Optimización por *recall*** en `GridSearchCV`: la función objetivo se alinea con minimizar FN, no
   con el accuracy global.

## 5. Limitaciones y próximos pasos

- **Tamaño y origen único:** 569 muestras de una sola institución (Wisconsin). La generalización a
  otras poblaciones/equipos de imagen debe validarse externamente.
- **Evaluación en una sola partición de test:** aunque se usó validación cruzada estratificada para la
  búsqueda de hiperparámetros, las métricas finales provienen de un único 20% de prueba. Próximo paso:
  **validación cruzada anidada** para estimar el desempeño con intervalos de confianza.
- **Umbral de decisión fijo en 0,5:** en un despliegue clínico real convendría **bajar el umbral** para
  reducir aún más los FN (a cambio de más FP), y reportar la curva *precision-recall*.
- **Próximos modelos:** contrastar con SVM y ensambles (Random Forest, Gradient Boosting) y añadir
  explicabilidad (SHAP) para auditar las predicciones individuales.
