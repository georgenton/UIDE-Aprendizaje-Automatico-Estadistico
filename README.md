# UIDE · Aprendizaje Automático Estadístico (MIA-B)

Repositorio del **componente práctico** de la asignatura *Aprendizaje Automático Estadístico* de la
**Maestría en Inteligencia Artificial Aplicada (MIA-B)** de la UIDE.

**Docente:** PhD. Iván García Santillán
**Grupo 2:** Jonathan Fabricio Gualli Ramírez · Jorge Armando Quizamánchuro Fuel · Raúl Marcelo Salazar Gamboa

---

## Contenido

| Semana | Tema | Estado |
|:------:|------|:------:|
| [Semana 1](semana-1/) | Clasificación de cáncer de mama (WBCD): Regresión Logística, KNN y Naive Bayes | ✅ |

## Entorno

```bash
python -m venv venv && source venv/bin/activate   # Python 3.12 (ver mise.toml)
pip install -r requirements.txt
```

## Semana 1 — Resumen de resultados

Clasificación binaria sobre el *Breast Cancer Wisconsin Diagnostic* (569 muestras, 30 variables).
Clase positiva = tumor **maligno**; métrica prioritaria = **Recall** (minimizar Falsos Negativos).

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

➡️ Detalle completo en [`semana-1/`](semana-1/README.md).
