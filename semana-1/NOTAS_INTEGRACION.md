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

> **Actualización (decisión final del equipo):** se decidió **adoptar la redacción de Raúl como base**
> del notebook de cáncer, corrigiendo únicamente erratas y cifras. Esta sección documenta cómo se hizo.
> El notebook se genera ahora con `patch_cancer.py` (toma el `.ipynb` de Raúl y le aplica los ajustes).

### a) Reescrituras de estilo → **se adoptaron, corrigiendo erratas y gramática**

La mayoría de sus cambios eran reescrituras de la redacción (acortar párrafos, quitar el encuadre
"*Qué haremos y por qué*"). **Se respetó su estilo**, corrigiendo solo lo que estaba mal escrito:

- erratas (*"anteriro"* → "anterior"; *"los 30 características"* → "las 30"; *"Analisis"* → "Análisis"),
- frases con gramática rota (la introducción de la Fase II: *"entender… nos permite"*), y
- una imprecisión conceptual (*"no vamos a quitar ninguna característica"* → "no eliminamos ninguna
  **fila**", que es lo correcto en el experimento de atípicos).

**Sobre la interpretación clínica FN/FP:** Raúl la había eliminado. Se **conservó** (sección 21) por
ser análisis sustantivo que premia la rúbrica, pero marcada con una nota para que el equipo decida si
retirarla y quedarse con la versión más breve de Raúl.

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
**coherencia entre secciones**). Se **conservó la redacción de las conclusiones de Raúl** pero con las
**cifras corregidas**, y **además** se añadió debajo un **bloque de "ranking consolidado" generado
automáticamente** desde `results_df`: así la narrativa es la de Raúl y las cifras quedan verificadas
contra la ejecución real, sin posibilidad de divergencia.

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
| Texto/redacción del compañero | **Se adopta**, respetando su estilo |
| Erratas o gramática rota dentro de esa redacción | Se **corrigen** (mínimo necesario) |
| Cifras que **contradicen la ejecución real** | Se corrigen; cuando es posible, se respaldan con un bloque generado automáticamente |
| Contenido analítico que el compañero eliminó (p. ej. FN/FP) | Se **conserva marcado**, para que el equipo decida |
| Diferencia que es solo un **matiz/empate** (Titanic) | Se respeta el texto y se añade una **nota** |

> Estas decisiones son **propuestas de integración**, no imposiciones. Conviene revisarlas en equipo
> antes de la entrega final.
