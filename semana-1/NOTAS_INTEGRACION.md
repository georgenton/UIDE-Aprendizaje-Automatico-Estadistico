# Notas de integración del equipo — Semana 1

Este documento deja constancia **transparente** de las decisiones tomadas al integrar los aportes de
los compañeros en los dos notebooks de la Semana 1, para que el equipo pueda revisarlas y discutirlas
antes de la entrega. El criterio rector fue uno solo: **preservar el trabajo de cada autor siempre que
sea correcto, y corregir únicamente lo que contradice la ejecución real del notebook.**

---

## 1. Notebook de Cáncer de Mama (WBCD) — versión enviada por Raúl

Raúl reenvió una versión del notebook de cáncer con cambios respecto al que tenía el equipo. Al
compararlos celda por celda se identificaron **tres tipos de cambios**, y se actuó distinto con cada
uno:

### a) Reescrituras de estilo → **NO se adoptaron** (se mantuvo la versión previa)

La mayoría de sus cambios eran reescrituras de la redacción (acortar párrafos, quitar el encuadre
"*Qué haremos y por qué*"). El problema es que varias de esas reescrituras:

- **introducían erratas** (p. ej. *"anteriro"*, *"los 30 características"*), y
- **eliminaban contenido con valor para la rúbrica**, en particular la **tabla de interpretación
  clínica de los errores (FN/FP)** y la sección de **Conclusiones** detallada.

**Por qué se mantuvieron las secciones originales:** la rúbrica premia la **argumentación y el
pensamiento crítico** y el ir *más allá de lo solicitado*. Quitar la lectura clínica de los Falsos
Negativos —el corazón del razonamiento médico del trabajo— habría restado puntos. Por eso se conservó
la narrativa más completa y sin erratas.

### b) Su aporte de fondo: una sección de "Conclusiones finales" → **se integró, pero corregida**

La idea de Raúl de cerrar con una sección de conclusiones consolidadas era **buena y se incorporó**.
Lo que **no** se pudo adoptar literalmente fueron sus **cifras**, porque **no coincidían con la
ejecución real** del notebook:

| Afirmación en el borrador de Raúl | Valor real (ejecución reproducible, `SEED=42`) |
|---|---|
| "recall de 0.9649" para el mejor modelo | Recall real = **0.9762** (Reg. Logística mejorada) |
| "reducir a **10** variables" | Selección real con \|r\|>0.4 = **20** variables de 30 |
| "KNN mejorado (F1 0.9615)" | F1 real de KNN mejorado = **0.9500** |

**Por qué se cambiaron los datos de las conclusiones:** entregar conclusiones con números que no
salen del propio notebook es un error grave de coherencia (la rúbrica evalúa explícitamente la
**coherencia entre secciones**). En lugar de copiar cifras escritas a mano —que además se
desactualizan en cuanto se reejecuta—, la sección de **"Conclusiones finales" se generó de forma
automática a partir de la tabla de resultados** (`results_df`). Así las cifras **siempre** reflejan la
ejecución real, sin posibilidad de divergencia.

### c) Cambió "tres" → "cuatro" algoritmos (anticipando Random Forest)

Raúl mencionó en el texto un cuarto algoritmo pero **no llegó a programarlo**. A pedido del equipo se
**implementó Random Forest** completo (baseline + mejorado con `GridSearchCV`, matrices de confusión,
curva ROC, importancia por Gini y modelo guardado en `joblib`), dejando el texto coherente con el
código.

> **Hallazgo honesto que surgió al añadir RF:** su *baseline* (Recall 0.929) **superó** a su versión
> optimizada (0.905) en el test. Se documentó como análisis crítico (el óptimo de validación cruzada
> no siempre transfiere al conjunto de prueba), en vez de ocultarlo.

---

## 2. Notebook del Titanic — versión enviada por otro compañero

Aquí la situación fue distinta y por eso **el criterio se aplicó al revés**: el notebook ya estaba
bien hecho, replicaba nuestro formato y sus conclusiones **sí eran coherentes** con su ejecución.

- Se **añadió Random Forest** (no lo tenía) y se alinearon las rutas a la convención del repo.
- Al sumar RF, el mejor modelo por **F1** pasó a ser *Random Forest baseline* (0.750) por delante del
  *KNN baseline* (0.748) que él citaba como ganador.
- **A pedido del equipo, NO se modificó su conclusión.** Se dejó **tal como la redactó** y se añadió
  únicamente una **celda de nota** con la sugerencia, sin tocar su texto.

**Por qué el trato fue diferente al del notebook de cáncer:** en cáncer las cifras del borrador
**contradecían** la ejecución (había que corregirlas por coherencia); en Titanic la conclusión del
autor era **correcta** y la diferencia es solo un matiz de empate técnico entre dos baselines, así que
bastó con una nota aclaratoria respetando su redacción.

---

## Resumen del criterio

| Situación | Acción |
|---|---|
| Texto del compañero **correcto** | Se mantiene tal cual |
| Reescritura que **introduce erratas o quita contenido valioso** | No se adopta; se conserva la versión más completa |
| Cifras que **contradicen la ejecución real** | Se corrigen (idealmente generándolas de forma automática) |
| Diferencia que es solo un **matiz/empate** | Se respeta el texto y se añade una **nota** |

> Estas decisiones son **propuestas de integración**, no imposiciones. Conviene revisarlas en equipo
> antes de la entrega final.
