# UIDE · Aprendizaje Automático Estadístico (MIA-B)

Repositorio del **componente práctico** de la asignatura *Aprendizaje Automático Estadístico* de la
**Maestría en Inteligencia Artificial Aplicada (MIA-B)** de la UIDE.

**Docente:** PhD. Iván García Santillán
**Grupo 2:** Jonathan Fabricio Gualli Ramírez · Jorge Armando Quizamánchuro Fuel · Raúl Marcelo Salazar Gamboa

---

## Contenido

| Semana | Tema | Estado |
|:------:|------|:------:|
| [Semana 1](semana-1/) | Clasificación supervisada — Cáncer de Mama (WBCD) y Titanic: Reg. Logística, KNN, Naive Bayes y Random Forest | ✅ |
| [Semana 2](semana-2/) | Titanic — 8 modelos (añade Árbol de Decisión, SVM, Gradient Boosting y Red Neuronal) | ✅ |
| [Semana 3](semana-3/) | WBCD y Titanic — Reducción de dimensionalidad (PCA/LDA) y agrupamiento (K-Means/GMM) | ✅ |

## Entorno

```bash
python -m venv venv && source venv/bin/activate   # Python 3.12 (ver mise.toml)
pip install -r requirements.txt
```

## Semana 1 — Resumen de resultados

Dos notebooks de clasificación binaria, cada uno con 4 algoritmos (baseline + mejorado).

### Cáncer de Mama (WBCD)

569 muestras, 30 variables. Clase positiva = tumor **maligno**; métrica prioritaria = **Recall**
(minimizar Falsos Negativos).

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

**Ganador:** Regresión Logística mejorada (mejor Recall y F1 = 0,976). **Random Forest** (propuesta del
equipo) es la mejor alternativa; curiosamente su *baseline* superó a su versión optimizada en el test.

### Titanic

891 pasajeros. Dataset "sucio" (nulos, categóricas, *feature engineering*) y más difícil; métrica
prioritaria = **F1** (costos de error simétricos).

| Modelo | Versión | Accuracy | Precision | Recall | F1 | AUC |
|--------|---------|:--------:|:---------:|:------:|:--:|:---:|
| Reg. Logística | Baseline | 0,8045 | 0,7833 | 0,6812 | 0,7287 | 0,8513 |
| KNN | Baseline | 0,8156 | 0,7903 | 0,7101 | 0,7481 | 0,8498 |
| Naive Bayes | Baseline | 0,7821 | 0,7143 | 0,7246 | 0,7194 | 0,8194 |
| **Random Forest** | **Baseline** | 0,8101 | 0,7612 | 0,7391 | **0,7500** | 0,8291 |

*(se muestran los baselines, los mejores; ver tabla completa de 8 filas en [`semana-1/`](semana-1/README.md))*

**Ganador:** Random Forest baseline (F1 = 0,750) ≈ KNN baseline (F1 = 0,748). Aquí **ninguna versión
mejorada superó al baseline**, coherente con un problema de señal débil.

➡️ Detalle completo en [`semana-1/`](semana-1/README.md).
