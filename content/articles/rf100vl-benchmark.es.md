---
title: "Benchmark RF100-VL: cómo generalizan los modelos LibreYOLO fuera de COCO"
description: Resultados de RF100-VL que miden cómo generalizan YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter y RF-DETR en 100 datasets del mundo real después del fine-tuning.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL mide hasta qué punto se adapta un detector más allá de COCO. Cada modelo se ajusta por separado con fine-tuning en 100 datasets muy distintos, incluidos datasets de infrarrojos, rayos X, microscopía, deportes, imágenes del espectro radioeléctrico, objetos diminutos y videojuegos. Evaluamos 17 configuraciones de LibreYOLO: **1700 fine-tunes en total.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Resultados

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L encabezó esta comparación con **61.76**. Le siguió M con 61.13 y después S con 60.41. Los cuatro resultados de RF-DETR son similares a las [cifras publicadas por Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/), lo que sirve como validación útil del entrenamiento de RF-DETR en LibreYOLO.

El tamaño no predijo todos los resultados. YOLO-NAS-S y M quedaron prácticamente empatados, con 58.00 y 57.99. EdgeCrafter-M obtuvo 57.93, por encima de S, con 55.99, y L, con 56.11. Tenemos previsto investigar estas regresiones. Este barrido no es un experimento de escalado controlado: varían la resolución, los pesos preentrenados, la precisión y las recetas.

RF-DETR-S con LoRA en el backbone obtuvo 57.19, frente a 60.41 con fine-tuning completo. El fine-tuning completo gana en 95 datasets. Según los registros, su mediana solo tarda siete minutos más, mientras que el tiempo total de los trabajos apenas cambia.

Las filas de YOLOv9 son anteriores a importantes correcciones de entrenamiento. El resultado anómalo de M ayudó a descubrir la ausencia de PGI, una convención distinta para letterbox y un límite de 100 etiquetas durante el entrenamiento. Las cifras archivadas describen el código que se ejecutó. No son un veredicto sobre la arquitectura YOLOv9.

Elige un modelo a continuación para consultar sus puntuaciones en los 100 datasets.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Metodología

Para cada dataset, entrenamos con `train`, seleccionamos el mejor checkpoint de AP50:95 en validación y lo evaluamos una vez en `test`. La puntuación principal es la media no ponderada de esos 100 resultados de test.

| Configuración | Regla de la campaña |
|---|---|
| Entrenamiento | 100 epochs; sin parada temprana |
| Batch efectivo | 16 |
| Seed | 0; una ejecución completada por dataset |
| Checkpoint | Mejor AP50:95 de validación usando pesos EMA |
| Puntuación de test | pycocotools con `maxDets=500` |
| Ajuste | Una receta fijada por familia; sin ajuste por dataset |

Cada familia mantuvo su propia receta de entrenamiento:

| Familia | Resolución | Precisión | Resumen del aumento de datos |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV y flip; aumentos intensos desactivados durante los últimos 15 epochs |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine y flip; tramo final de 15 epochs |
| YOLOX S/M | 640 | FP32 | La misma receta de la familia, con un learning rate específico para cada tamaño |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV y flip; mosaic desactivado |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; valores predeterminados de la familia LibreYOLO |
| RF-DETR N/S/M/L y S LoRA | 384/512/576/704; LoRA a 512 | BF16 | Multi-scale, crop/resize y flip |

Estos resultados corresponden a una sola seed. No se ha demostrado que las diferencias pequeñas sean mejoras. RF-DETR M y L usaron acumulación de gradientes para ajustarse a la memoria disponible, y algunos datasets densos necesitaron batches físicos más pequeños. Además, las recetas públicas de RF-DETR parten de checkpoints de COCO, mientras que Roboflow usó checkpoints privados de Objects365 para su tabla histórica.

Los tiempos de entrenamiento son registros de la campaña, no benchmarks de velocidad controlados. En ocasiones, los trabajos compartieron GPU, se reintentaron o se reanudaron. No ejecutamos el protocolo opcional de latencia de un único artefacto en T4. YOLO-NAS usa los [pesos preentrenados con licencia independiente para uso no comercial de Deci](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).

## Mejoras derivadas de la carga de trabajo

Las pruebas pequeñas pueden demostrar que el entrenamiento se inicia. No reproducen semanas de fine-tuning continuo en muchas arquitecturas y datasets. A esta escala, los caminos lentos, la presión de memoria y los problemas de corrección se hicieron difíciles de pasar por alto.

- **Carga y validación de imágenes.** Almacenamos en caché el redimensionamiento determinista antes del aumento de datos, reutilizamos los workers y buffers del validador, y generamos gráficos de las pasadas de validación cuando era posible. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **CUDA graphs durante el entrenamiento.** La compatibilidad con la captura se amplió de YOLOv9 y RF-DETR a 24 familias. También corregimos una race condition de pin-memory en DataLoader y la invalidación de los graphs durante las transiciones del entrenamiento. Una prueba de YOLOv9-T pasó de 428.4 a 367.7 segundos con el mismo AP registrado. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **Puntuación COCO.** La evaluación de datasets densos a veces tardaba más que el entrenamiento. La integración faster-coco-eval puntuó las predicciones guardadas de las 100 particiones de test en 8.4 segundos, en lugar de 131.4. De 1,400 valores de métricas, 1,381 fueron idénticos bit a bit; la mayor diferencia fue 2.22e-16 y el AP principal no cambió. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR y rutas de entrenamiento compartidas.** Una búsqueda L1 más rápida, menos sincronizaciones de dispositivo, AdamW fusionado y una memoria acotada para el matcher redujeron un paso de RF-DETR-S de 266 a 234 ms en la prueba registrada. La misma investigación mejoró otros cinco matchers, el logging de YOLOv9 y la reutilización de tensores de EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Atención.** Redirigimos la atención compatible a PyTorch SDPA, integramos una implementación CUDA con licencia Apache-2.0 y corregimos la ejecución con precisión mixta. También escribimos un kernel Triton de atención deformable dentro del repositorio para inferencia. En las pruebas documentadas, fue unas cuatro veces más rápido de forma aislada y alrededor de un 7% más rápido de extremo a extremo para RF-DETR-N. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Corrección del entrenamiento.** Corregimos la reconstrucción de BatchNorm de YOLOX y restauramos PGI, los metadatos de letterbox, la capacidad de etiquetas y el warmup del momentum de YOLOv9. El benchmark proporcionó los checkpoints y los patrones de fallo que dejaron al descubierto ambos problemas. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Esas cifras proceden de pruebas independientes con distintas cargas de trabajo. No deben multiplicarse para presentar una única mejora de velocidad. LibreYOLO ahora es más rápido y fiable para cargas de trabajo de entrenamiento reales que antes de esta campaña.

## Harness y artefactos

El [harness público de entrenamiento](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) programa los datasets entre GPU, reanuda los trabajos interrumpidos, gestiona la recuperación de OOM y registra las recetas, las versiones de los datasets, los logs, los checkpoints y las predicciones.

La [skill run-rf100vl-benchmark](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) proporciona a los agentes de programación con IA el protocolo y las instrucciones de operación de Vast.ai. El [archivo de resultados](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) contiene todas las ejecuciones publicadas.

## Gracias, Roboflow

Joseph Nelson ofreció soporte de GPU después de que escribiera que quería evaluar más allá de COCO, pero no podía permitirme las ejecuciones. Matvei Popov compartió las configuraciones de referencia, explicó los ajustes de evaluación y siguió los resultados a medida que llegaban.

Ese apoyo nos permitió validar el entrenamiento de LibreYOLO en 17 configuraciones, publicar evidencia útil para elegir un modelo, publicar el harness y someter la biblioteca a pruebas de estrés hasta que sus puntos débiles quedaron a la vista.

Gracias a Joseph, Matvei y al equipo de Roboflow por los recursos de cómputo, el tiempo y la orientación.
