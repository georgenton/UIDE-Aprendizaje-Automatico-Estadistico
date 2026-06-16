"""
Parchea S1_Titanic_Clasificacion.ipynb (notebook del compañero) para:
  1) Alinear rutas a la convención del notebook de cáncer (models/outputs/data en semana-1/).
  2) Añadir Random Forest como 4º algoritmo (baseline + mejorado, matrices, ROC, importancia Gini).
  3) Añadir sección dinámica de "Conclusiones finales" (misma idea que en cáncer).
  4) Ajustar numeración de secciones y textos que quedarían inconsistentes.
Conserva intacto el resto del trabajo original del compañero.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "semana-1" / "notebook" / "S1_Titanic_Clasificacion.ipynb"
nb = nbf.read(NB_PATH, as_version=4)


def replace_in_cell(marker, old, new, required=True):
    for c in nb.cells:
        src = "".join(c["source"])
        if marker in src and old in src:
            c["source"] = src.replace(old, new)
            return True
    if required:
        raise SystemExit(f"NO encontrado: marker={marker!r} old={old[:40]!r}")
    return False


# ---------------------------------------------------------------
# 1) Rutas alineadas al esquema de cáncer
# ---------------------------------------------------------------
replace_in_cell(
    "MODELS_DIR",
    '# Rutas relativas (el notebook vive en la raíz del proyecto, junto a los .csv)\n'
    'NB_DIR = Path.cwd()\n'
    'MODELS_DIR = NB_DIR / "models"\n'
    'OUT_DIR = NB_DIR / "outputs"\n'
    'MODELS_DIR.mkdir(exist_ok=True)\n'
    'OUT_DIR.mkdir(exist_ok=True)',
    '# Rutas relativas desde semana-1/notebook (mismo esquema que el notebook de cáncer)\n'
    'NB_DIR = Path.cwd()\n'
    'PROJ_DIR = NB_DIR.parent           # semana-1/\n'
    'MODELS_DIR = PROJ_DIR / "models"\n'
    'OUT_DIR = PROJ_DIR / "outputs"\n'
    'DATA_DIR = PROJ_DIR / "data"\n'
    'MODELS_DIR.mkdir(exist_ok=True)\n'
    'OUT_DIR.mkdir(exist_ok=True)',
)

replace_in_cell(
    "CSV_PATH",
    'CSV_PATH = NB_DIR / "titanic_train.csv"\n'
    'if not CSV_PATH.exists():               # respaldo por si el nombre cambia\n'
    '    CSV_PATH = NB_DIR / "titanic.csv"',
    'CSV_PATH = DATA_DIR / "titanic_train.csv"\n'
    'if not CSV_PATH.exists():               # respaldo por si el nombre cambia\n'
    '    CSV_PATH = DATA_DIR / "titanic.csv"',
)

# ---------------------------------------------------------------
# 2) Textos de consistencia (tres -> cuatro, título, tabla)
# ---------------------------------------------------------------
replace_in_cell("Fase III", "Entrenamos tres clasificadores clásicos",
                "Entrenamos cuatro clasificadores")
replace_in_cell("# Clasificación de Supervivencia",
                "Regresión Logística, KNN y Naive Bayes",
                "Regresión Logística, KNN, Naive Bayes y Random Forest")
replace_in_cell("Tabla comparativa", "Consolidamos las seis filas (3 modelos × 2 versiones).",
                "Consolidamos las ocho filas (4 modelos × 2 versiones).")
replace_in_cell("comparativa_metricas", "t17_comparativa_metricas.png",
                "t21_comparativa_metricas.png")

# Observación de la tabla: al añadir RF, el mejor por F1 pasa a ser RF baseline (antes KNN baseline).
replace_in_cell(
    "Observación honesta",
    "> **Observación honesta:** el mejor desempeño en el test lo da el **KNN baseline**; ninguna versión\n"
    "> *mejorada* supera a su *baseline* en esta partición. Se discute por qué en las Conclusiones.\n"
    "\n"
    "**Observación:** el mejor desempeño en el test lo da el **KNN baseline**; ninguna versión\n"
    "*mejorada* supera a su *baseline* en esta partición. Se discute por qué en las Conclusiones.",
    "> **Observación honesta:** el mejor desempeño en el test se lo reparten dos *baselines* casi\n"
    "> empatados: **Random Forest baseline** (mejor **F1 = 0.750**, la métrica que priorizamos) y\n"
    "> **KNN baseline** (mejor Accuracy = 0.816 y AUC). **Ninguna versión *mejorada* supera a su\n"
    "> *baseline*** en esta partición. Se discute por qué en las Conclusiones.",
)

# Conclusiones: incluir RF entre los que superan a NB, y actualizar el "mejor modelo".
replace_in_cell(
    "Dataset más difícil",
    "4. **Regresión Logística y KNN** superan a **Naive Bayes**, cuyo supuesto de independencia y\n"
    "   normalidad se ve penalizado por las variables categóricas/discretas del Titanic.",
    "4. **Regresión Logística, KNN y Random Forest** superan a **Naive Bayes**, cuyo supuesto de\n"
    "   independencia y normalidad se ve penalizado por las variables categóricas/discretas del Titanic.",
)
replace_in_cell(
    "Dataset más difícil",
    "5. **La versión *mejorada* no superó al *baseline* en el test.** El mejor modelo fue **KNN baseline**\n"
    "   (F1=0.748, Accuracy=0.816). Es un resultado legítimo y esperable: con pocas variables y señal débil,\n"
    "   el baseline ya opera cerca del techo del problema, y ni el ajuste de hiperparámetros ni la selección\n"
    "   de características transfieren mejora a esta partición.",
    "5. **La versión *mejorada* no superó al *baseline* en ninguno de los cuatro modelos.** Por F1 (la\n"
    "   métrica priorizada) el mejor fue **Random Forest baseline** (F1=0.750, el 4º modelo propuesto por\n"
    "   el equipo), seguido muy de cerca por **KNN baseline** (F1=0.748, y el mejor en Accuracy=0.816 y\n"
    "   AUC): están prácticamente empatados. Es un resultado legítimo y esperable: con pocas variables y\n"
    "   señal débil, el baseline ya opera cerca del techo del problema, y ni el ajuste de hiperparámetros\n"
    "   ni la selección de características transfieren mejora a esta partición.",
)

# Artefacto RF en el dict de guardado
replace_in_cell(
    "artifacts = {",
    '    "titanic_naive_bayes_model.pkl":         {"model": nb_best,      "features": selected_features, "scaler": "StandardScaler"},\n}',
    '    "titanic_naive_bayes_model.pkl":         {"model": nb_best,      "features": selected_features, "scaler": "StandardScaler"},\n'
    '    "titanic_random_forest_model.pkl":       {"model": rf_best,      "features": selected_features, "scaler": "StandardScaler (no requerido por RF)"},\n}',
)

# Renumeración de secciones posteriores (insertamos RF como 20 y conclusiones como 23)
replace_in_cell("Guardado de los modelos", "## 20. Guardado de los modelos (joblib)",
                "## 21. Guardado de los modelos (joblib)")
replace_in_cell("Tabla comparativa", "## 21. Tabla comparativa final de todos los modelos",
                "## 22. Tabla comparativa final de todos los modelos")
replace_in_cell("¿qué error priorizar", "## 22. Interpretación: ¿qué error priorizar en el Titanic?",
                "## 24. Interpretación: ¿qué error priorizar en el Titanic?")

# ---------------------------------------------------------------
# 3) Construcción de celdas nuevas
# ---------------------------------------------------------------
def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))

def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))

rf_cells = [
    md(r"""
## 20. Modelo 4 — Random Forest (propuesta del equipo)

**Por qué lo añadimos:** a propuesta del equipo incorporamos un modelo de **ensamble basado en
árboles**. Random Forest combina muchos árboles entrenados sobre submuestras aleatorias (*bagging*) y
subconjuntos de variables; capta **interacciones no lineales** (p. ej. la combinación sexo × clase ×
edad, el famoso "mujeres y niños de primera clase primero") que un modelo lineal no representa de forma
directa, y entrega su propia **importancia de características** (impureza de Gini).

> **Nota:** los árboles son **invariantes a la escala**, por lo que no necesitan normalización.
> Reutilizamos las matrices ya escaladas por **coherencia** con el resto del flujo; esto no altera el
> resultado de un bosque.

**Baseline:** por defecto, todas las variables. **Mejorada:** `GridSearchCV` sobre `n_estimators`,
`max_depth`, `min_samples_split` y `max_features`, optimizando **F1** (igual que el resto de modelos de
este notebook), con selección de características.
"""),
    code(r"""
from sklearn.ensemble import RandomForestClassifier

# --- Baseline ---
rf_base = RandomForestClassifier(random_state=SEED).fit(X_train_std, y_train)
row_b, pred_b, proba_b = evaluate(rf_base, X_test_std, y_test, "Random Forest", "Baseline")
print("BASELINE:", {k: round(v, 4) for k, v in row_b.items() if isinstance(v, float)})

# --- Mejorada: GridSearchCV + selección de características ---
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=SEED), param_grid_rf,
                       scoring="f1", cv=cv, n_jobs=-1)
grid_rf.fit(X_train_std[:, sel_idx], y_train)
rf_best = grid_rf.best_estimator_
row_m, pred_m, proba_m = evaluate(rf_best, X_test_std[:, sel_idx], y_test, "Random Forest", "Mejorado")
print("Mejores hiperparámetros:", grid_rf.best_params_)
print("MEJORADO:", {k: round(v, 4) for k, v in row_m.items() if isinstance(v, float)})
"""),
    code(r"""
plot_confusion(y_test, pred_b, "Random Forest — Baseline", "t17_cm_rf_baseline.png")
plot_confusion(y_test, pred_m, "Random Forest — Mejorado", "t18_cm_rf_mejorado.png")
plot_roc([("Baseline", y_test, proba_b), ("Mejorado", y_test, proba_m)],
         "Curva ROC — Random Forest", "t19_roc_rf.png")
"""),
    md(r"""
### Importancia de características (Random Forest, Gini) — análisis adicional

Random Forest mide la importancia por la **reducción de impureza (Gini)** que aporta cada variable a lo
largo del bosque. Contrastarla con los coeficientes lineales permite ver si ambos enfoques coinciden en
señalar a `Sex`, `Pclass` y `Fare` como los grandes determinantes de la supervivencia.
"""),
    code(r"""
rf_imp = pd.Series(rf_best.feature_importances_, index=selected_features).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=rf_imp.values, y=rf_imp.index, palette="viridis", ax=ax)
ax.set_title("Importancia de características — Random Forest (mejorado, Gini)")
ax.set_xlabel("Importancia (reducción media de impureza)")
plt.tight_layout(); plt.savefig(OUT_DIR / "t20_importancia_rf.png", dpi=120); plt.show()
"""),
    md(r"""
**Comparación baseline vs mejorado (Random Forest):** el ensamble suele rendir muy bien ya en
*baseline*; la búsqueda de hiperparámetros regula profundidad y número de árboles para equilibrar
sesgo-varianza. Por ser un modelo de mayor capacidad entrenado sobre pocas variables, conviene vigilar
el sobreajuste. Si su importancia Gini coincide con los coeficientes lineales en destacar `Sex`,
`Pclass` y `Fare`, es una buena señal de que el modelo aprende patrones históricamente plausibles.
"""),
]

conclu_cells = [
    md(r"""
## 23. Conclusiones finales (resumen consolidado)

Consolidamos las conclusiones de los **cuatro** algoritmos. El bloque siguiente las **deriva
automáticamente** de la tabla de resultados (la métrica prioritaria en este problema es **F1**, por la
simetría de costos), de modo que las cifras siempre coinciden con la ejecución real.
"""),
    code(r"""
ranking = results_df.sort_values(["F1", "AUC", "Accuracy"], ascending=False).reset_index(drop=True)
win = ranking.iloc[0]

print("CONCLUSIONES FINALES  (métrica prioritaria en este problema = F1)")
print("=" * 68)
print(f"1) Mejor modelo: {win['Modelo']} ({win['Versión']})")
print(f"   F1={win['F1']:.4f} | Recall={win['Recall']:.4f} | "
      f"Precision={win['Precision']:.4f} | AUC={win['AUC']:.4f} | Accuracy={win['Accuracy']:.4f}")
print(f"\n2) Selección de características: {len(selected_features)} de {len(feature_cols)} "
      f"variables (|r| > {THRESHOLD}).")
print(f"   Top-4 más predictivas: {', '.join(selected_features[:4])}.")
print(f"\n3) Ranking completo por F1 (desempate por AUC y Accuracy):")
for i, r in ranking.iterrows():
    print(f"   {i+1}. {r['Modelo']:<16} {r['Versión']:<9} "
          f"F1={r['F1']:.4f}  AUC={r['AUC']:.4f}  Acc={r['Accuracy']:.4f}")

mejoras = []
for modelo in results_df["Modelo"].unique():
    sub = results_df[results_df["Modelo"] == modelo].set_index("Versión")["F1"]
    mejoras.append((modelo, sub["Mejorado"] - sub["Baseline"]))
n_mejora = sum(1 for _, d in mejoras if d > 1e-9)
peor = [m for m, d in mejoras if d < -1e-9]
print(f"\n4) La optimización (GridSearchCV + selección) mejoró el F1 en {n_mejora} de {len(mejoras)} modelos.")
if peor:
    print(f"   En {', '.join(peor)} el baseline igualó o superó a la versión ajustada en el test: con "
          f"un dataset 'difícil' y pocas variables, más optimización NO garantiza mejor generalización. "
          f"Lección: medir siempre en un test intacto.")
"""),
]

# ---------------------------------------------------------------
# 4) Insertar las celdas nuevas en las posiciones correctas
# ---------------------------------------------------------------
def index_of(substr):
    for i, c in enumerate(nb.cells):
        if substr in "".join(c["source"]):
            return i
    raise SystemExit(f"No se encontró celda con {substr!r}")

# RF: después de la celda de plots de Naive Bayes (la que guarda t16_roc_nb.png)
i_nb_plots = index_of("t16_roc_nb.png")
nb.cells[i_nb_plots + 1:i_nb_plots + 1] = rf_cells

# Conclusiones finales: después de la celda del gráfico comparativo (t21_comparativa_metricas.png)
i_comp = index_of("t21_comparativa_metricas.png")
nb.cells[i_comp + 1:i_comp + 1] = conclu_cells

nbf.write(nb, NB_PATH)
print(f"Notebook parcheado: {NB_PATH} ({len(nb.cells)} celdas)")
