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
