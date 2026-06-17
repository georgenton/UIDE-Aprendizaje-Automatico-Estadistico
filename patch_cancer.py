"""
Parchea el notebook de CÁNCER tomando como base la versión redactada por RAÚL
(S1_Cancer_Mama_Clasificacion.ipynb). A pedido del equipo:
  - Se ADOPTA la redacción de Raúl, corrigiendo únicamente erratas y cifras incorrectas.
  - Se MANTIENE Random Forest (4º algoritmo) y la sección de conclusiones finales dinámica.
  - Se conserva la interpretación clínica FN/FP (análisis sustantivo; marcado para revisión del equipo).
Reemplaza al antiguo build_notebook.py como generador de este notebook.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "semana-1" / "notebook" / "S1_Cancer_Mama_Clasificacion.ipynb"
nb = nbf.read(NB_PATH, as_version=4)


def replace_in_cell(marker, old, new, required=True):
    for c in nb.cells:
        src = "".join(c["source"])
        if marker in src and old in src:
            c["source"] = src.replace(old, new)
            return True
    if required:
        raise SystemExit(f"NO encontrado: marker={marker!r} old={old[:50]!r}")
    return False


def set_cell(marker, new_source):
    for c in nb.cells:
        if marker in "".join(c["source"]):
            c["source"] = new_source.strip("\n")
            return True
    raise SystemExit(f"set_cell: no encontrado {marker!r}")


def index_of(substr):
    for i, c in enumerate(nb.cells):
        if substr in "".join(c["source"]):
            return i
    raise SystemExit(f"index_of: no encontrado {substr!r}")


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))

def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


# ===============================================================
# 1) CORRECCIÓN DE ERRATAS Y GRAMÁTICA (redacción de Raúl)
# ===============================================================
replace_in_cell("Resumen del trabajo",
                "a partir de los 30 características",
                "a partir de las 30 características")

# Fase III en el resumen: incluir Random Forest (se mantiene como 4º modelo)
replace_in_cell("Resumen del trabajo",
                "**Regresión Logística**, **KNN** y **Naive Bayes**, cada uno",
                "**Regresión Logística**, **KNN**, **Naive Bayes** y **Random Forest** "
                "(este último a propuesta del equipo), cada uno")

# Bullet de importancia: añadir lectura Gini (RF se mantiene)
replace_in_cell("Análisis adicionales",
                "- **Importancia de características** de manera visual a partir de los datos de la Regresión Logística.",
                "- **Importancia de características** de manera visual: coeficientes (Reg. Logística) "
                "e impureza Gini (Random Forest).")

# Fase II intro: corregir gramática ("entender ... nos permite")
replace_in_cell("Fase II — Análisis Exploratorio de Datos (EDA)",
                "Necesitamos entender la estructura de los datos antes de modelar nos permite detectar\n"
                "desbalance de clases, identificar las variables más informativas y justificar la selección de\n"
                "características con criterio estadístico.",
                "Entender la estructura de los datos antes de modelar nos permite detectar el desbalance de\n"
                "clases, identificar las variables más informativas y justificar la selección de\n"
                "características con criterio estadístico.")

# Histogramas: cerrar la frase correctamente
replace_in_cell("Histogramas de las variables",
                "linealmente, es decir no existe solapamiento",
                "linealmente; es decir, con poco solapamiento entre clases.")

# Títulos con "Analisis" -> "Análisis"
replace_in_cell("StandardScaler y MinMaxScaler",
                "## 12. Analisis con StandardScaler y MinMaxScaler",
                "## 12. Análisis con StandardScaler y MinMaxScaler")
replace_in_cell("Eliminar o conservar",
                "## 13. Analisis para Eliminar o conservar los atípicos:",
                "## 13. Análisis: eliminar o conservar los atípicos")

# Decisión escalador: errata "anteriro" + redacción
replace_in_cell("normalizador base",
                "**Considerando el resultado anteriro, nos fijamos que el recall es idéntico con ambos "
                "escaladores. Se adopta StandardScaler como normalizador base por ser el más apropiado para "
                "los modelos de distancia (KNN) y lineales.**",
                "**Considerando el resultado anterior, el recall es idéntico con ambos escaladores. Se "
                "adopta StandardScaler como normalizador base por ser el más apropiado para los modelos de "
                "distancia (KNN) y lineales.**")

# Decisión atípicos: redacción ("quitar ninguna característica" -> filas) + "~30%"
replace_in_cell("observamos que el recall",
                "**Considerando el resultado anterior, observamos que el recall (prioridad clínica) es "
                "IDÉNTICO en ambas estrategias; eliminar solo mejora de forma marginal accuracy/F1 en ESTA "
                "partición, a costa de descartar 30% del entrenamiento y la señal de los tumores más "
                "agresivos. Por lo que no vamos a realizar quitar ninguna característica inicial**",
                "**Considerando el resultado anterior, el recall (prioridad clínica) es IDÉNTICO en ambas "
                "estrategias; eliminar solo mejora de forma marginal accuracy/F1 en ESTA partición, a costa "
                "de descartar ~30% del entrenamiento y la señal de los tumores más agresivos. Por eso "
                "conservamos todos los datos: no eliminamos ninguna fila.**")

# ===============================================================
# 2) RE-NUMERACIÓN + FIGURA COMPARATIVA + ARTEFACTO RF
# ===============================================================
replace_in_cell("Guardado de los modelos", "## 18. Guardado de los modelos",
                "## 19. Guardado de los modelos")
replace_in_cell("Tabla comparativa", "## 19. Tabla comparativa final de todos los modelos",
                "## 20. Tabla comparativa final de todos los modelos")
replace_in_cell("comparativa_metricas", "16_comparativa_metricas.png",
                "20_comparativa_metricas.png")
replace_in_cell("artifacts = {",
                '    "naive_bayes_model.pkl":         {"model": nb_best,      "features": selected_features, "scaler": "StandardScaler"},\n}',
                '    "naive_bayes_model.pkl":         {"model": nb_best,      "features": selected_features, "scaler": "StandardScaler"},\n'
                '    "random_forest_model.pkl":       {"model": rf_best,      "features": selected_features, "scaler": "StandardScaler (no requerido por RF)"},\n}')

# ===============================================================
# 3) CORRECCIÓN DE CIFRAS EN LAS CONCLUSIONES DE RAÚL
# ===============================================================
set_cell("## Conclusiones finales", r"""
## Conclusiones finales

- El mejor modelo en recall es la **Regresión Logística mejorada** (con selección de características e
  hiperparámetros optimizados), que alcanza un **recall de 0.976**, superando a su baseline (0.929) y a
  los demás algoritmos.
- La Regresión Logística también lidera en **F1 (0.976)**; le sigue el KNN mejorado (F1 0.95) y, como
  mejor alternativa, el **Random Forest**. Naive Bayes es el más bajo (recall 0.857), penalizado por su
  supuesto de independencia frente a la fuerte multicolinealidad.
- La selección de características por correlación retuvo **20 variables** (|r| > 0.4), mejorando la
  interpretabilidad y reduciendo la multicolinealidad. Las 4 más predictivas fueron `worst concave
  points`, `worst perimeter`, `mean concave points` y `worst radius`.
- **Random Forest** (4.º modelo, propuesta del equipo) aporta una segunda lectura de importancia (Gini)
  coherente con esas variables; un hallazgo instructivo es que su *baseline* superó a su versión
  optimizada en el test.
- En base a la matriz de confusión y la curva ROC, la Regresión Logística mejorada detecta mejor los
  malignos (recall) sin sacrificar precisión, lo que la convierte en la opción más robusta para este
  caso clínico.
""")

# ===============================================================
# 3b) APLICACIÓN PROFESIONAL — reflexiones reales de los 3 integrantes
# ===============================================================
APLICACION = r"""
---
## 🔧 Aplicación Profesional — Learning by Doing

> *Cada integrante del equipo describe en 2-3 líneas cómo aplicaría los contenidos de esta semana en su
> entorno laboral o proyecto de titulación.*

**Jorge Armando Quizamánchuro Fuel:**
En mi plataforma SaaS de psicoeducación, este flujo de clasificación binaria con priorización del
recall es directamente aplicable a un *screening* de riesgo psicológico: igual que aquí el Falso
Negativo (no detectar un maligno) es el error crítico, en salud mental no detectar a un usuario en
riesgo es el error que más debemos minimizar. La selección de características por correlación y la
comparación baseline vs mejorado me dan un marco reproducible para construir modelos interpretables y
auditables sobre datos de cuestionarios de bienestar.

**Jonathan Fabricio Gualli Ramírez:**
Desde mi lado de DevOps/desarrollo, lo que me quedó claro esta semana es que el modelo no acaba en el
`.fit()`: acaba cuando se puede reproducir y desplegar. Que fijar `random_state` dé siempre el mismo
resultado es lo mismo que busco en un *build* de CI/CD, y guardar el modelo en `.pkl` es tener un
artefacto que puedo meter en una API y versionar. La idea de baseline vs mejorado me sirve como un
filtro de calidad: solo sube la versión que mejora. Me veo automatizando este ciclo con un pipeline
simple de MLOps en mi trabajo.

**Raúl Marcelo Salazar Gamboa:**
En mi ámbito profesional, el desarrollo de nuevos productos bancarios, puedo aplicar la regresión
logística para mejorar los sistemas de puntuación (*scoring*) de cuentas colectoras, estatales y
corrientes, mejorando la precisión en la evaluación del perfil de riesgo de los clientes jurídicos e
institucionales. Esto permitirá optimizar la predictibilidad de comportamientos inusuales, mitigar el
riesgo de contraparte y agilizar los tiempos de aprobación mediante un modelo automatizado, robusto y
fácilmente auditable por los entes reguladores.

---
"""
set_cell("Aplicación Profesional", APLICACION)

# ===============================================================
# 4) CELDAS NUEVAS — Random Forest (recall), interpretación clínica y ranking dinámico
# ===============================================================
rf_cells = [
    md(r"""
## 18. Modelo 4 — Random Forest (propuesta del equipo)

**Por qué lo añadimos:** incorporamos un modelo de **ensamble basado en árboles**. Random Forest
combina muchos árboles entrenados sobre submuestras aleatorias (*bagging*) y subconjuntos de variables;
capta interacciones no lineales y entrega su propia **importancia de características** (impureza de
Gini), útil para contrastar con los coeficientes lineales.

> **Nota:** los árboles son **invariantes a la escala**, por lo que no necesitan normalización.
> Reutilizamos las matrices ya escaladas por coherencia con el resto del flujo.

**Baseline:** por defecto, todas las variables. **Mejorada:** `GridSearchCV` sobre `n_estimators`,
`max_depth`, `min_samples_split` y `max_features`, optimizando **recall**, con selección de variables.
"""),
    code(r"""
from sklearn.ensemble import RandomForestClassifier

# --- Baseline ---
rf_base = RandomForestClassifier(random_state=SEED).fit(X_train_std, y_train)
row_b, pred_b, proba_b = evaluate(rf_base, X_test_std, y_test, "Random Forest", "Baseline")
print("BASELINE:", {k: round(v, 4) for k, v in row_b.items() if isinstance(v, float)})

# --- Mejorada ---
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=SEED), param_grid_rf,
                       scoring="recall", cv=cv, n_jobs=-1)
grid_rf.fit(X_train_std[:, sel_idx], y_train)
rf_best = grid_rf.best_estimator_
row_m, pred_m, proba_m = evaluate(rf_best, X_test_std[:, sel_idx], y_test, "Random Forest", "Mejorado")
print("Mejores hiperparámetros:", grid_rf.best_params_)
print("MEJORADO:", {k: round(v, 4) for k, v in row_m.items() if isinstance(v, float)})
"""),
    code(r"""
plot_confusion(y_test, pred_b, "Random Forest — Baseline", "16_cm_rf_baseline.png")
plot_confusion(y_test, pred_m, "Random Forest — Mejorado", "17_cm_rf_mejorado.png")
plot_roc([("Baseline", y_test, proba_b), ("Mejorado", y_test, proba_m)],
         "Curva ROC — Random Forest", "18_roc_rf.png")
"""),
    md(r"""
### Importancia de características (Random Forest, Gini) — análisis adicional

Random Forest mide la importancia por la **reducción de impureza (Gini)**. Contrastarla con los
coeficientes lineales enriquece la interpretabilidad del diagnóstico.
"""),
    code(r"""
rf_imp = pd.Series(rf_best.feature_importances_, index=selected_features).sort_values()
fig, ax = plt.subplots(figsize=(8, 7))
sns.barplot(x=rf_imp.values, y=rf_imp.index, palette="viridis", ax=ax)
ax.set_title("Importancia de características — Random Forest (mejorado, Gini)")
ax.set_xlabel("Importancia (reducción media de impureza)")
plt.tight_layout(); plt.savefig(OUT_DIR / "19_importancia_rf.png", dpi=120); plt.show()
"""),
    md(r"""
**Comparación baseline vs mejorado (Random Forest):** el ensamble rinde muy bien ya en *baseline*; de
hecho su baseline supera a la versión optimizada en recall sobre el test. Es un recordatorio de que el
óptimo hallado por validación cruzada no siempre transfiere al conjunto de prueba.
"""),
]

interp_clinica = md(r"""
## 21. Interpretación clínica: por qué minimizar los Falsos Negativos

> *Sección de análisis FN/FP conservada del trabajo del equipo (puede retirarse si se prefiere la
> versión más breve de Raúl).*

En la matriz de confusión, con **maligno = clase positiva (1)**:

| Celda | Significado clínico |
|-------|---------------------|
| **TP** (Verdadero Positivo) | Tumor maligno correctamente detectado ✅ |
| **TN** (Verdadero Negativo) | Tumor benigno correctamente identificado ✅ |
| **FP** (Falso Positivo) | Benigno clasificado como maligno → ansiedad y pruebas adicionales (costo **moderado**) |
| **FN** (Falso Negativo) | **Maligno clasificado como benigno → diagnóstico perdido, tratamiento retrasado (costo GRAVE)** ❌ |

**El Falso Negativo es el error más peligroso.** Por eso la métrica prioritaria es el **Recall /
Sensibilidad** $= \frac{TP}{TP+FN}$. Un modelo con accuracy alto pero recall bajo sería **clínicamente
inaceptable**; en un despliegue real incluso se bajaría el **umbral de decisión** para reducir aún más
los FN.
""")

ranking_cells = [
    md(r"""
**Ranking consolidado (generado automáticamente desde los resultados, para garantizar coherencia entre
las cifras citadas y la ejecución real):**
"""),
    code(r"""
ranking = results_df.sort_values(["Recall", "F1", "AUC"], ascending=False).reset_index(drop=True)
win = ranking.iloc[0]
print("CONCLUSIONES FINALES  (prioridad clínica = Recall de la clase maligna)")
print("=" * 70)
print(f"1) Mejor modelo: {win['Modelo']} ({win['Versión']})")
print(f"   Recall={win['Recall']:.4f} | F1={win['F1']:.4f} | "
      f"Precision={win['Precision']:.4f} | AUC={win['AUC']:.4f}")
print(f"\n2) Variables en las versiones mejoradas: {len(selected_features)} de {len(feature_cols)} "
      f"(|r| > {THRESHOLD}). Top-4: {', '.join(selected_features[:4])}.")
print(f"\n3) Ranking por Recall (desempate por F1 y AUC):")
for i, r in ranking.iterrows():
    print(f"   {i+1}. {r['Modelo']:<16} {r['Versión']:<9} "
          f"Recall={r['Recall']:.4f}  F1={r['F1']:.4f}  AUC={r['AUC']:.4f}")
mejoras = []
for modelo in results_df["Modelo"].unique():
    sub = results_df[results_df["Modelo"] == modelo].set_index("Versión")["Recall"]
    mejoras.append((modelo, sub["Mejorado"] - sub["Baseline"]))
n_mejora = sum(1 for _, d in mejoras if d > 1e-9)
peor = [m for m, d in mejoras if d < -1e-9]
print(f"\n4) La optimización mejoró el recall en {n_mejora} de {len(mejoras)} modelos.")
if peor:
    print(f"   EXCEPCIÓN instructiva: {', '.join(peor)} — su baseline superó a la versión ajustada en "
          f"el test (el óptimo de validación cruzada no siempre transfiere).")
"""),
]

# ===============================================================
# 5) INSERCIONES
# ===============================================================
# RF: después de la celda de plots de Naive Bayes (15_roc_nb.png)
i = index_of("15_roc_nb.png")
nb.cells[i + 1:i + 1] = rf_cells

# Interpretación clínica: justo antes de la conclusión de Raúl
i = index_of("## Conclusiones finales")
nb.cells[i:i] = [interp_clinica]

# Ranking dinámico: justo después de la conclusión de Raúl
i = index_of("## Conclusiones finales")
nb.cells[i + 1:i + 1] = ranking_cells

nbf.write(nb, NB_PATH)
print(f"Notebook (base Raúl) parcheado: {NB_PATH} ({len(nb.cells)} celdas)")
