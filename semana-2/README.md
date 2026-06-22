# Semana 2 — Titanic: 8 modelos de clasificación

**Práctica grupal — Semana 2**
**Materia:** Aprendizaje Automático Estadístico — MIA-B
**Docente:** PhD. Iván García Santillán
**Grupo 2:** Jonathan Fabricio Gualli Ramírez · Jorge Armando Quizamánchuro Fuel · Raúl Marcelo Salazar Gamboa

---

## Objetivo

Aplicar el **mismo flujo de 3 fases** del notebook de Cáncer de Mama (de Marcelo) al dataset del
**Titanic** (predicción de supervivencia), comparando **8 modelos** en versión *baseline* y *mejorada*
(`GridSearchCV` / `RandomizedSearchCV`, `scoring="recall"`):

Regresión Logística · KNN · Naive Bayes · Random Forest · Árbol de Decisión · SVM ·
Gradient Boosting · **Red Neuronal (Keras)**.

> **Clase positiva = `survived = 1`.** Prioridad: **Recall** (en rescate, el Falso Negativo es el
> error más costoso).

> **Nota:** este repositorio contiene **solo el notebook del Titanic** (responsabilidad del equipo). El
> notebook de Cáncer de Mama de la Semana 2 lo entrega Marcelo por separado.

## Dataset

`sns.load_dataset('titanic')` (copia local en `data/titanic_seaborn.csv`). 891 pasajeros, 342
sobrevivieron / 549 no (~38% / 62%). El Titanic exige **mucho preprocesamiento** (nulos, codificación,
eliminación de columnas con fuga como `alive`), que es su valor didáctico.

## Estructura

```
semana-2/
├── notebook/S2_Titanic_Clasificacion.ipynb   # Notebook completo y ejecutado (8 modelos)
├── models/                                     # best_*.joblib + 2 redes .keras
├── data/titanic_seaborn.csv
├── outputs/                                    # figuras + resultados_comparativa.csv
└── README.md
```

## Decisión metodológica documentada

El notebook de cáncer seleccionaba variables con **|r| > 0.4** (WBCD tiene correlaciones > 0.7). En el
Titanic las correlaciones son más débiles (`sex` ≈ 0.54, `pclass` ≈ −0.34, `fare` ≈ 0.26): con 0.4 solo
pasaría `sex`. Se adoptó **|r| > 0.2** → `sex`, `pclass`, `fare`, coherente con que *el Titanic es un
problema menos separable linealmente*.

## Resultados (versiones Mejoradas, ordenadas por Recall)

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|--------|:--------:|:---------:|:------:|:--:|:---:|
| **Red Neuronal** | 0,788 | 0,725 | **0,725** | **0,725** | **0,859** |
| Naive Bayes | 0,760 | 0,681 | 0,710 | 0,695 | 0,801 |
| Random Forest | 0,771 | 0,700 | 0,710 | 0,705 | 0,823 |
| Gradient Boosting | 0,782 | 0,727 | 0,696 | 0,711 | 0,836 |
| KNN | 0,777 | 0,730 | 0,667 | 0,697 | 0,799 |
| Reg. Logística | 0,777 | 0,730 | 0,667 | 0,697 | 0,820 |
| SVM | 0,782 | 0,742 | 0,667 | 0,702 | 0,806 |
| Árbol de Decisión | 0,771 | 0,719 | 0,667 | 0,692 | 0,768 |

**Mejor modelo:** la **Red Neuronal** mejorada (mejor Recall, F1 y AUC).

## Conclusiones clave

- La **variable más predictiva es el sexo** ("mujeres y niños primero") — el ML revela un sesgo
  histórico en los datos.
- El Titanic es **más difícil** que el WBCD: métricas en torno a 0,75–0,80 frente al ~0,95–0,98 del
  cáncer, por la menor correlación lineal de las variables con el objetivo.
- **Naive Bayes** sufre por su supuesto de independencia (correlación `pclass`/`fare`).
- En desbalance, el **Recall** importa más que el Accuracy: priorizamos no dejar pasar sobrevivientes.

📓 [Ver notebook](notebook/S2_Titanic_Clasificacion.ipynb)
