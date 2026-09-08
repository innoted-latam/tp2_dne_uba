# Revisión de resultados y plan de mejora (Bloques 17+)

## 1. Dónde quedamos

Cuatro experimentos ejecutados (bloques 7-16), todos con ResNet50 congelada
como extractor fijo + cabeza densa entrenable:

| Experimento | Accuracy | Recall clase 1 |
|---|---|---|
| 1 frame, 10 épocas fijas | 0.57 | 0.34 |
| 1 frame, early stopping | 0.53 | 0.16 |
| 1 frame, augmentation + ES | 0.54 | 0.52 |
| 4 frames, augmentation + ES (promedio) | 0.58 | 0.24 |

Baseline trivial (predecir siempre clase mayoritaria): **0.561**.

Pendientes de la consigna: GradCAM (interpretabilidad) y conclusiones.

## 2. El diagnóstico real: el problema es de MEDICIÓN, no de modelo

El test tiene **132 secuencias**. Con n=132 y accuracy ~0.55, el error
estándar es 0.043, o sea un intervalo de confianza del 95% de **±8.5 puntos**.

| Experimento | Accuracy | IC 95% |
|---|---|---|
| 1 frame, 10 épocas | 0.57 | [0.486, 0.654] |
| 1 frame, early stopping | 0.53 | [0.445, 0.615] |
| 4 frames, promedio | 0.58 | [0.496, 0.664] |

Consecuencias, ambas importantes:

1. **Los cuatro experimentos son estadísticamente indistinguibles entre sí.**
   La diferencia entre el "mejor" (0.58) y el "peor" (0.53) tiene z = 0.82
   (p ≈ 0.41). No hubo mejora ni empeoramiento: hubo ruido de muestreo.
2. **El IC de todos incluye al baseline 0.561.** No se puede afirmar que
   ninguno supere al baseline, pero tampoco que no lo supere.

Es decir: las decisiones que se venían tomando (augmentation "mejoró",
early stopping "empeoró", 4 frames "es lo mejor") se apoyaban en diferencias
que el experimento no tiene poder para detectar. **Antes de probar modelos
nuevos hay que arreglar cómo se mide**, o vamos a seguir persiguiendo ruido.

## 3. Segundo problema: la métrica

`accuracy` con un umbral fijo de 0.5 es la peor métrica posible acá:
- Es discreta y de alta varianza en muestras chicas.
- Depende del umbral, que nunca se calibró. Por eso el recall de la clase 1
  salta entre 0.16 y 0.52 sin que el modelo cambie sustancialmente: lo que se
  mueve es dónde cae el umbral, no cuánta señal hay.
- **ROC-AUC** es independiente del umbral y detecta señal débil que accuracy
  esconde. Un modelo con AUC 0.62 tiene señal real aunque su accuracy sea 0.56.

Además se está tirando información: `memorability_score` es continuo y se lo
binariza a `memorable`. Entrenar **regresión sobre el score** aprovecha mucho
más el dataset chico, y permite reportar **correlación de Spearman**, que es
la métrica oficial del desafío MediaEval original.

## 4. Tercer problema (posible): fuga de datos por película

`sequence_name` = nombre de película + segundo de inicio + segundo de fin.
Varias secuencias vienen de la **misma película**. El split actual es por
secuencia, así que frames de una misma película pueden caer a ambos lados
del split: el modelo puede reconocer la película (paleta, iluminación,
actores) en vez de aprender memorabilidad. Hay que medir cuántas películas
distintas hay y, si son pocas, **splitear por película**.

## 5. Plan de mejora, ordenado por impacto

| # | Acción | Por qué |
|---|---|---|
| A | **Precomputar embeddings** de ResNet50 una sola vez (2640 × 2048) | Con el backbone congelado los features nunca cambian: recalcularlos en cada época es puro costo. Precomputar convierte experimentos de minutos en segundos y habilita todo lo demás. |
| B | **Validación cruzada 5-fold** sobre las 660 secuencias + reportar **AUC** | Ataca el problema central: usa el 100% de los datos para evaluar en vez del 20%, y da desvío estándar entre folds. Recién ahí se puede decir si algo mejora. |
| C | **Regresión sobre `memorability_score`** + umbral calibrado en validación | Usa más información que la etiqueta binaria y permite reportar Spearman como el desafío original. |
| D | **Un vector por secuencia** (promedio/concatenación de los 4 frames) en vez de promediar predicciones | Agrega en el espacio de features, no en el de decisiones; elimina las filas duplicadas con la misma etiqueta. |
| E | **Split por película** y comparar contra el split por secuencia | Cuantifica la fuga del punto 4. |
| F | **Comparar backbones** (ResNet50 vs EfficientNetB0 vs MobileNetV2) | Con embeddings precomputados cuesta minutos y es una comparación honesta para el informe. |
| G | **Fine-tuning** del último bloque conv, LR 1e-5 + augmentation | Último recurso: con 528 secuencias el riesgo de overfitting es alto. Solo tiene sentido *después* de tener una medición confiable. |
| H | **GradCAM + conclusiones** | Requerido explícitamente por la consigna. |

## 6. Expectativa realista de resultados

El desafío original (MediaEval 2019) se resuelve con datasets de 8.000-10.000
videos y el estado del arte llega a Spearman ≈ 0.5. Acá hay **660 secuencias**.
Un resultado honesto y bien medido en el rango **AUC 0.60-0.68** es un buen
trabajo; forzar una accuracy alta con un test de 132 casos sería reportar ruido.

El valor del TP no está en el número final sino en: medir bien, diagnosticar
por qué el modelo no aprende más, y documentarlo. El punto 2 de este
documento es, en sí mismo, un hallazgo que vale la pena escribir en las
conclusiones.

## 7. Nota sobre "solo técnicas vistas en clase"

La cabeza que se propone entrenar sobre los embeddings (regresión logística)
es **matemáticamente idéntica** a la `Dense(1, sigmoid)` que ya se venía
usando: mismo modelo, distinta implementación. `scikit-learn` se usa solo
para que la validación cruzada sea una línea de código. Si se prefiere,
la misma cabeza se puede escribir en Keras sin cambiar nada del resultado.


---

# Resultados de la ejecución (bloques 17a-17f)

## Lo que se ejecutó

| Bloque | Qué | Resultado |
|---|---|---|
| 17a | Embeddings ResNet50 (`pooling='avg'`) de los 2640 frames | matriz 2640 × 2048 |
| 17b | Promedio por secuencia + labels | 660 × 2048, baseline 0.561 |
| 17c | 5-fold CV, regresión logística (`C=0.01`) | acc 0.579 ± 0.022 — **AUC 0.601 ± 0.028** |
| 17d | Conteo de películas en `sequence_name` | **98 películas**, ~7 secuencias cada una |
| 17e | Mismo modelo, `StratifiedGroupKFold` por película | acc 0.541 ± 0.037 — **AUC 0.548 ± 0.041** |
| 17f | Regresión (Ridge, `alpha=1000`) vs clasificación, `GroupKFold` | clasif. AUC 0.574 / Spearman 0.163 — **regr. AUC 0.593 / Spearman 0.207** |

## Análisis

**1. La fuga por película valía entre 3 y 5 puntos de AUC.** 0.601 sin agrupar
vs 0.548 agrupado sugiere 5.3 puntos, pero la misma clasificación agrupada dio
0.574 con otro splitter (`GroupKFold` en vez de `StratifiedGroupKFold`). El
número agrupado se mueve entre 0.548 y 0.574 según el particionado: la fuga es
real y grande —aproximadamente la mitad de la señal aparente— pero su magnitud
exacta tiene una barra de error que estos datos no permiten achicar.

**2. La regresión sobre el score continuo gana poco pero gana en ambas
métricas** (+0.019 AUC, +0.044 Spearman, mismo splitter). La coincidencia de
dirección en dos métricas es lo que le da credibilidad. **No se testeó
significancia**: haría falta comparación pareada fold a fold. Es "consistente",
no "probado".

**3. Accuracy por debajo del baseline con AUC por encima del azar** (0.541 vs
0.561, y 0.548 vs 0.500) no es contradictorio: es umbral mal calibrado. El
modelo ordena algo mejor que el azar, pero cortar en 0.5 rinde peor que
predecir siempre la clase mayoritaria. Confirma que accuracy nunca debió ser
la métrica principal.

**4. El desvío entre folds creció de ±0.028 a ±0.041 al agrupar.** Cada fold
pasa de 132 secuencias sueltas a ~20 películas: menos unidades efectivas,
estimación más ruidosa. De acá en adelante solo son detectables mejoras de
varios puntos.

**5. Hallazgo de fondo: el tamaño efectivo del dataset no es 660, es ~98.**
Las 7 secuencias de una misma película no son observaciones independientes.
Esto explica el techo mejor que cualquier detalle de arquitectura, y explica
por qué ninguno de los experimentos de los bloques 7-16 movía la aguja.

Contexto: el desafío MediaEval original alcanza Spearman ~0.5 con 8.000-10.000
videos. Spearman 0.207 con 98 películas es proporcionado.

## Cierre recomendado

- **Bloque 18 — GradCAM** (requerido por la consigna). Necesita un modelo Keras
  con la CNN adentro; usar el del bloque 12, no el Ridge sobre embeddings.
- **Bloque 19 — tabla final y conclusiones.** El eje del informe no es el AUC
  alcanzado sino que los resultados previos estaban inflados por dos sesgos
  independientes (test chico, fuga por película), invisibles sin cambiar el
  esquema de evaluación.
- **Bloque 17g (opcional)** — comparar `EfficientNetB0` como extractor, para
  cubrir la pregunta "¿probaste otro backbone?". No se espera que cambie nada.
- **Descartado: fine-tuning.** Con ~98 unidades independientes el overfitting
  es casi seguro. Se argumenta en las conclusiones, no se prueba.

---

# Estado final (bloques 17g-19)

| Bloque | Qué | Resultado |
|---|---|---|
| 17g | EfficientNetB0 como extractor, mismo pipeline | AUC 0.594 / Spearman 0.190 (vs ResNet50: 0.593 / 0.207) |
| 18 | GradCAM sobre el modelo del bloque 12 | Los mapas se encienden sobre luminancia y contraste, no sobre contenido semántico. Acierta 1 de 4 ejemplos; predicciones comprimidas entre 0.21 y 0.60 |
| 19 | Tabla comparativa final + conclusiones | — |

**Modelo final:** Ridge sobre embeddings de ResNet50 congelada, agregados por
secuencia, evaluado con `GroupKFold` agrupado por película.
**ROC-AUC 0.593, Spearman 0.207.**

Que dos backbones distintos (ResNet50 y EfficientNetB0) topen en el mismo AUC
es la evidencia más fuerte de que el cuello de botella son los datos (~98
películas independientes) y no la arquitectura. Junto con GradCAM —que muestra
al modelo respondiendo a propiedades fotométricas de bajo nivel— cierra el
argumento para descartar el fine-tuning.

TP cerrado. El notebook (`tp2_dne_uba.ipynb`, 98 celdas con outputs) está en
este repo.
