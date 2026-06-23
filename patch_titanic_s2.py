"""
Parchea el notebook del Titanic (Semana 2) tomando como base la versión CORREGIDA por Marcelo
(S2_Titanic_Clasificacion.ipynb de Descargas). Incorpora sus dos pedidos:
  1) Añade el gráfico del Árbol de Decisión (plot_tree) en el Modelo 5.
  2) Cambia la arquitectura BASELINE de la Red Neuronal a una sola capa densa de 3 neuronas.
  3) Corrige una errata ("solo sobre el train y no sobre el train").
Reemplaza a build_titanic_s2.py como generador de este notebook.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "semana-2" / "notebook" / "S2_Titanic_Clasificacion.ipynb"
nb = nbf.read(NB_PATH, as_version=4)


def replace_in_cell(marker, old, new):
    for c in nb.cells:
        s = "".join(c["source"])
        if marker in s and old in s:
            c["source"] = s.replace(old, new)
            return True
    raise SystemExit(f"NO encontrado: marker={marker!r} old={old[:50]!r}")


def index_of(substr):
    for i, c in enumerate(nb.cells):
        if substr in "".join(c["source"]):
            return i
    raise SystemExit(f"index_of: no encontrado {substr!r}")


def md(t): return nbf.v4.new_markdown_cell(t.strip("\n"))
def code(t): return nbf.v4.new_code_cell(t.strip("\n"))


# 1) Corregir errata del experimento de outliers
replace_in_cell("máscara IQR",
                "**solo sobre el train** y no sobre el train.",
                "**solo sobre el train**, nunca sobre el test.")

# 2) Red Neuronal BASELINE -> una sola capa densa de 3 neuronas (pedido de Marcelo)
replace_in_cell("Arquitectura Baseline",
                "**Arquitectura Baseline:** Dense(16, relu) → Dropout(0.2) → Dense(1, sigmoid).",
                "**Arquitectura Baseline:** una sola **capa densa de 3 neuronas** — Dense(3, relu) → "
                "Dense(1, sigmoid). Es el punto de partida (siguiendo la fórmula indicada por el "
                "profesor) desde el cual se mejora la arquitectura.")
replace_in_cell("nn_base = build_nn",
                "nn_base = build_nn([(16, 0.2)])",
                "nn_base = build_nn([(3, 0.0)])   # baseline: 1 capa densa de 3 neuronas")

# 3) Gráfico del Árbol de Decisión (plot_tree) — punto requerido de la práctica
tree_cells = [
    md(r"""
### Visualización del Árbol de Decisión

Una de las ventajas del Árbol de Decisión es su **interpretabilidad gráfica**: podemos ver las reglas
que aprende. Mostramos los primeros 3 niveles del árbol mejorado.
"""),
    code(r"""
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 8))
plot_tree(dt_best, max_depth=3, feature_names=selected_features,
          class_names=["No Sobrevivió", "Sobrevivió"], filled=True, rounded=True, fontsize=9)
plt.title("Árbol de Decisión (primeros 3 niveles)")
plt.tight_layout(); plt.savefig(OUT_DIR / "m5_dt_tree.png", dpi=120); plt.show()
"""),
]
i = index_of("best_dtree_model.joblib")
nb.cells[i + 1:i + 1] = tree_cells

# 4) Nota aclaratoria tras la nueva red base (no se modifica la conclusión del autor)
nota = md(r"""
> **Nota (tras incorporar la red neuronal base de 3 neuronas):** al reejecutar el notebook, la **Red
> Neuronal Mejorada** quedó como la mejor también en **Accuracy** (además de Recall, F1 y AUC). El
> punto 2 (Gradient Boosting con mejor Accuracy) corresponde a una ejecución previa; las redes
> neuronales presentan cierta variabilidad entre corridas. La conclusión del autor se conserva tal cual.
""")
i = index_of("El mejor modelo por el recall")
nb.cells[i + 1:i + 1] = [nota]

nbf.write(nb, NB_PATH)
print(f"Notebook (base Marcelo) parcheado: {NB_PATH} ({len(nb.cells)} celdas)")
