# TP2 - M72109 (Datos No Estructurados, UBA-FCE)

## Contexto del trabajo

Notebook de Colab: `tp2_dne_uba.ipynb` (vive en Google Drive del usuario,
título `tp2_dne_uba.ipynb`, id de Drive `1qz7yjaRDUqsQSsXpTbcfS76wdrC8aO5n`).
Repo Git (`innoted-latam/tp2_dne_uba`) tiene una copia, puede estar desactualizada.

**Consigna:** predecir si un video es "memorable" (`memorable`: 0/1) a partir
de frames extraídos, usando técnicas de Computer Vision vistas en la materia.
Debe usar solo técnicas vistas en clase, ser ejecutable de punta a punta,
documentado paso a paso.

**Material de referencia leído y resumido** (dos archivos subidos por el
usuario, ya analizados en profundidad):
- Módulo NLP (`datos_no_estructurado.md`) — no aplica directo al TP2 (es de texto).
- Módulo Visión/Video (`BIBLIO_IMAGENYVIDEO.md`) — la fuente real de técnicas
  para este TP2: CNN desde cero, transfer learning (ResNet50 vía
  `tensorflow.keras.applications`, congelando pesos), data augmentation
  (`RandomFlip/RandomRotation/RandomZoom`), GradCAM (interpretabilidad),
  video como secuencia de frames (CNN+LSTM con `TimeDistributed`).

## Datos

- `dataset_df`: 2640 filas, columnas `sequence_name`, `image_path`, `memorable`,
  `memorability_score`. Cada secuencia de video tiene 4 frames: `_0`, `_56`
  (nombrado `56` en archivos reales, no `52`), `_112`, `_histogram`
  (en inglés, NO `histograma`).
- Naming real de archivos: `<sequence_name>__<frame>.jpg` (doble guion bajo).
- Balance de clases: 56.1% `memorable=0`, 43.9% `memorable=1` (leve, no
  requiere balanceo).

## Metodología acordada con el usuario

- Usuario es principiante: SIEMPRE bloque de código corto → usuario lo corre
  en Colab → pasa el resultado → yo analizo → genero texto breve en markdown
  para pegar en una celda de texto en Colab debajo del bloque. Nunca avanzar
  sin ver el resultado real.
- Modo **ponytail** activo (`/ponytail full`): respuestas cortas, sin relleno,
  código mínimo, explicación breve.
- Bloques grandes se subdividen en a/b/c/d cuando conviene.

## Progreso: bloques ya ejecutados y resultados

**Bloque 1 (EDA):** value_counts + imágenes de muestra. Balance 56/44, sin
patrón visual obvio entre clases.

**Bloque 2 (split train/test):** split por `sequence_name` (NO por fila, para
evitar data leakage entre frames de la misma secuencia). 80/20 estratificado
por `memorable`. Resultado: 528 secuencias train (2112 filas), 132 test
(528 filas), intersección 0.

**Bloque 3 (selección de frame):** se decidió arrancar con un solo frame por
secuencia: `histogram` (el marcado como más informativo). Filtro correcto:
`str.contains('__histogram')`. Resultado: 528 train / 132 test, 1 imagen por
secuencia.

**Bloque 4 (pipeline tf.data):** función `load_image` (decode_jpeg + resize a
224x224). `train_ds`/`test_ds` batch=32. Verificado shape (32,224,224,3),
píxeles en rango [0,255] (normalización se hace después, en el modelo).

**Bloque 5 (modelo, subdividido a/b/c/d):**
- 5a: `ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))`
  — 23.587.712 params, salida (7,7,2048).
- 5b: `base_model.trainable = False` → 0 params entrenables.
- 5c: modelo = `Lambda(preprocess_input)` + base_model + `GlobalAveragePooling2D`
  + `Dense(64, relu)` + `Dense(1, sigmoid)`.
- 5d: summary total 23.718.913 params, solo 131.201 entrenables (0.55%).

**Bloque 6 (compilación):** `optimizer='adam'`, `loss='binary_crossentropy'`,
`metrics=['accuracy']`.

**Bloque 7 (entrenamiento, 10 épocas fijas, SIN early stopping):**
train accuracy 0.54→0.84, val accuracy estancada 0.55-0.59. **Overfitting
claro** desde época 0 en val.

**Bloque 8 (curvas):** confirma overfitting. Loss de val toca mínimo en
época 3 (~0.69), después empeora.

**Bloque 9 (evaluación test, modelo de 10 épocas):** accuracy 0.57
(baseline trivial ~0.56). Recall clase 1 (memorable): 0.34 — sesgado a
predecir "no memorable".

**Bloque 10 (early stopping, intento 1 — CON BUG):** se reentrenó SIN
reconstruir el modelo desde cero → siguió entrenando desde los pesos ya
sobreajustados del bloque 7 (arrancó en accuracy 0.85). Bug detectado y
corregido.

**Bloque 10 (corregido, modelo fresco + early stopping):** paró en época 6,
restauró pesos de época 3 (val_loss mínimo 0.699, val_accuracy 0.53).

**Bloque 11 (evaluación, early stopping limpio):** accuracy 0.53 (peor que
bloque 9). Recall clase 1 cayó a 0.16 — modelo colapsó prediciendo casi
siempre "no memorable". Conclusión: el cuello de botella no es el momento
de frenar el entrenamiento, sino que ResNet50 congelada + 1 frame no captura
señal suficiente.

**Decisión del usuario:** probar data augmentation antes que fine-tuning o
multi-frame (motivo: fine-tuning es más riesgoso con dataset chico de 528
secuencias).

**Bloque 12 (modelo fresco + augmentation + early stopping):**
`data_augmentation = Sequential([RandomFlip('horizontal'), RandomRotation(0.1),
RandomZoom(0.1)])`, insertado antes de `preprocess_input`. Paró en época 6,
restauró época 3 (val_loss 0.687, val_accuracy 0.538) — leve mejora vs sin
augmentation.

**Bloque 13 (evaluación con augmentation):** accuracy 0.54 (similar), pero
predicciones balanceadas: recall clase 0 = 0.55, clase 1 = 0.52 (F1 clase 1
subió de 0.23 a 0.50). Ya no colapsa hacia una clase. Accuracy sigue sin
superar baseline de forma clara.

**Decisión del usuario:** probar usar los 4 frames por secuencia (en vez de
fine-tuning), agregando predicciones por secuencia (promedio), reusando la
misma arquitectura (menos riesgo que descongelar ResNet50 con dataset chico).

**Bloque 14 (dataset con 4 frames):** `train_ds_all`/`test_ds_all` usando
`train_df`/`test_df` completos (todas las filas, no solo histogram). Train
2112 filas (528 secuencias × 4), test 528 filas (132×4).

**Bloque 15 (reentrenamiento, mismo modelo con augmentation, sobre 4 frames):**
paró en época 5, restauró época 2 (val_loss 0.701, val_accuracy 0.549).
Overfitting persiste igual que con 1 frame — hipótesis: los 4 frames de una
misma secuencia son muy parecidos entre sí, no aportan tanta diversidad como
4 videos distintos.

**Bloque 16 (evaluación agregando 4 predicciones por secuencia, promedio de
`y_prob` agrupado por `sequence_name`):** accuracy 0.58 (la más alta de
todos los experimentos), pero recall clase 1 volvió a caer a 0.24 (sesgo
hacia clase mayoritaria).

## Tabla comparativa de resultados (hasta ahora)

| Experimento | Accuracy | Recall clase 1 (memorable) |
|---|---|---|
| 1 frame, 10 épocas fijas | 0.57 | 0.34 |
| 1 frame, early stopping | 0.53 | 0.16 |
| 1 frame, augmentation + early stopping | 0.54 | 0.52 (mejor balance) |
| 4 frames, augmentation + early stopping (promedio) | 0.58 | 0.24 |

**Ningún experimento supera claramente el baseline trivial (~0.56, predecir
siempre la clase mayoritaria).** Diagnóstico: con ResNet50 congelada como
extractor de features fijo, el modelo no encuentra señal fuerte para
memorabilidad, independientemente de cuántos frames o cuánto se regularice.

## Punto de decisión (revisado)

Ver `PLAN_MEJORA.md` — incluye los resultados ya ejecutados de los bloques 17a-17f. Resumen:

**Hallazgo central de la revisión:** el test tiene solo 132 secuencias, lo
que da un IC 95% de ±8.5 puntos sobre accuracy. Los cuatro experimentos
(0.53 / 0.54 / 0.57 / 0.58) son **estadísticamente indistinguibles entre sí**
(diferencia máxima: z = 0.82, p ≈ 0.41) y todos tienen al baseline 0.561
dentro de su intervalo. Las conclusiones intermedias que se venían sacando
("augmentation mejoró", "early stopping empeoró", "4 frames es lo mejor")
estaban leyendo ruido de muestreo. **Antes de probar modelos nuevos hay que
arreglar la medición.**

Orden de trabajo acordado (bloques 17+):
A. Precomputar embeddings de ResNet50 (backbone congelado ⇒ features fijos).
B. Validación cruzada 5-fold sobre las 660 secuencias + métrica ROC-AUC.
C. Regresión sobre `memorability_score` + umbral calibrado (y Spearman).
D. Agregar los 4 frames en el espacio de features, no de predicciones.
E. Verificar/evitar fuga por película (`sequence_name` incluye el film).
F. Comparar backbones (ResNet50 / EfficientNetB0 / MobileNetV2).
G. Fine-tuning del último bloque conv (último recurso).
H. GradCAM + conclusiones (requeridos por la consigna).

Bloques restantes del plan original (11 bloques definidos al inicio, ahora
ampliados a ~17 por las iteraciones): falta cerrar con GradCAM
(interpretabilidad, pedido explícitamente por la consigna: "documentar el
paso a paso de por qué realiza lo que realiza") y conclusiones finales.

## Reglas de flujo para retomar

- Seguir dando bloques cortos de código, uno a la vez.
- Esperar que el usuario corra en Colab y pegue el resultado real antes de
  avanzar.
- Después de cada resultado, dar un texto breve en markdown para pegar en
  Colab como celda de texto.
- Modo ponytail: sin relleno, código mínimo, explicaciones cortas.
- No asumir resultados, no inventar salidas — siempre esperar el output real
  del usuario.
