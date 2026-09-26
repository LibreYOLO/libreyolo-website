---
title: Cuantización
seo_title: Cuantizar un modelo LibreYOLO en PyTorch
description: >-
  La API de cuantización de LibreYOLO en PyTorch: nueve recetas, la calibración
  separada de los datos de entrenamiento, QAT y QAD, y dos artefactos de
  despliegue.
lead: >-
  La cuantización en LibreYOLO se ejecuta enteramente en PyTorch:
  model.quantize() sustituye los módulos Conv2d y Linear de un modelo por
  equivalentes cuantizados y los calibra. El resultado mantiene el contrato
  habitual de predict, val, train y save, así que un modelo cuantizado se puntúa
  con los mismos validadores que uno en float.
keywords:
  - cuantización libreyolo
  - ptq int8 pytorch
  - entrenamiento consciente de la cuantización
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - dataset de calibración
  - exportar onnx qdq
last_verified: 1.5.0
meta:
  - label: Llamada
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Comando
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: Extra
    value: Ninguno. La cuantización se ejecuta en PyTorch.
  - label: Familias
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: Recetas
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: Artefactos de despliegue
    value: >-
      export(format="pt") para un checkpoint empaquetado, export(format="onnx")
      para un grafo QDQ INT8
    mono: true
verification: >-
  Leído de libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py y docs/quantization.md en la rama dev. Las
  cifras de tamaño de los checkpoints son los valores medidos que se registran
  en docs/quantization.md.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Intercambio de estructura más calibración. calib es un pequeño
        conjunto

        # de imágenes SIN ETIQUETAR, leído solo hacia delante para derivar los

        # rangos de activación y las escalas.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # los mismos validadores que un
        modelo float

        qmodel.save("LibreYOLO9s-int8.pt")     # el checkpoint lleva un
        manifiesto quant
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: Argumentos
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # ruta a un data.yaml o nombre integrado; None omite la calibración
            samples=128,               # máximo de imágenes de calibración
            batch=8,                   # tamaño de batch de calibración
            algorithm="auto",          # auto y minmax son lo mismo; percentile es la alternativa
            keep_high_precision=None,  # None usa la política de la familia
            verbose=True,
        )
  reload:
    - label: Un checkpoint cuantizado se recarga como tal
      language: python
      code: |
        from libreyolo import LibreYOLO

        # El manifiesto quant reconstruye la estructura cuantizada y las escalas
        # antes de que se carguen los pesos.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT es un train() normal sobre un modelo cuantizado
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Un fine-tuning, no una ejecución desde cero: usa learning rates de
        fine-tuning.

        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD añade los argumentos de destilación existentes
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: Checkpoint de PyTorch empaquetado
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Escribe LibreYOLO9s-int8-final.pt: pesos y escalas empaquetados en
        baja

        # precisión, maestros fp32 eliminados, resto no cuantizado convertido a
        fp16.

        qmodel.export(format="pt")


        # remainder="fp32" mantiene exactos los tensores no cuantizados.

        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Pares QuantizeLinear/DequantizeLinear dentro del grafo que llevan las
        # escalas calibradas o entrenadas con QAT del propio modelo.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 'Vuelta a float, conservando los pesos entrenados con QAT'
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        qmodel.dequantize()


        # Ahora sirve cualquier exportador float, en cualquier precisión que
        soporte.

        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## Instalación

La cuantización no necesita ningún extra. El intercambio de módulos, la pasada de
calibración y la aritmética simulada se ejecutan todos en PyTorch, así que
`pip install libreyolo` es todo el requisito. Los artefactos de despliegue
necesitan lo que necesite su propio formato, que en el caso de la ruta ONNX es
`libreyolo[onnx]`.

## Cuantización

<code-tabs name="quantize" />

`quantize()` transforma el modelo cargado in situ y lo devuelve. No hay
gradientes de por medio: el intercambio instala módulos cuantizados y la pasada
de calibración se ejecuta solo hacia delante.

El checkpoint resultante es un checkpoint normal de LibreYOLO con un manifiesto
`quant` adjunto, así que se recarga con su estructura y sus escalas intactas:

<code-tabs name="reload" />

Los checkpoints que el entrenador escribe durante una ejecución de QAT llevan
también el manifiesto, lo que significa que el `best.pt` de una ejecución así es
en sí mismo un checkpoint cuantizado.

## Recetas

Se admiten cuatro familias: `yolo9`, `rfdetr`, `birefnet` y `feynobg`.

| Receta | Qué hace | Familias | Calibración |
|---|---|---|---|
| `fp16` | Cast a media precisión con un contrato de entrada y salida en float32. Solo inferencia. | las cuatro | ninguna |
| `bf16` | Cast a bfloat16, que conserva el rango de exponente de float32. La solución cuando fp16 se desborda en un modelo de tipo DETR. Solo inferencia. | las cuatro | ninguna |
| `fp8` | Pesos y activaciones E4M3 en `Conv2d` y `Linear`: escalas de peso por canal, escalas de activación por tensor calibradas. | las cuatro | requerida |
| `int8` | W8A8 en `Conv2d` y `Linear`: pesos simétricos por canal, activaciones afines por tensor. | las cuatro | requerida, o `calib=None` para cuantizar solo los pesos |
| `w4a16` | Pesos INT4 simétricos agrupados, grupo de 128 a lo largo de `in_features`, activaciones en float, en `Linear`. | rfdetr, birefnet, feynobg | no hace falta |
| `w4a8` | Pesos INT4 agrupados más activaciones INT8 calibradas, en `Linear`. | rfdetr, birefnet, feynobg | requerida |
| `nvfp4` | NVFP4 W4A4 en `Linear`: elementos E2M1, bloques de 16 elementos, escalas de bloque FP8 E4M3, escala de tensor FP32. Escalado dinámico de activaciones. | rfdetr, birefnet, feynobg | no hace falta |
| `mxfp4` | MXFP4 de OCP en `Linear`: elementos E2M1, bloques de 32 elementos, escalas de bloque E8M0 en potencias de dos. Escalado dinámico de activaciones. | rfdetr, birefnet, feynobg | no hace falta |
| `int2` | Solo investigación: pesos agrupados de 2 bits, grupo de 64, más activaciones INT8, en `Linear`. Solo con post-entrenamiento es inservible, así que hace falta QAT o QAD. | rfdetr | requerida |

Las recetas por debajo de 8 bits apuntan a `nn.Linear` y se rechazan a propósito
para `yolo9`: en el hardware actual esa aceleración es solo para GEMM, así que
las convoluciones se quedan en mayor precisión. YOLO9 usa `int8` o `fp8`. `int2`
se rechaza para `birefnet` y `feynobg` porque esas familias son solo de
inferencia, así que ahí no está disponible el QAT de recuperación del que depende
la receta.

Los valores por defecto de cada familia mantienen la primera capa y las cabezas
en float, y la convolución DFL de YOLO9 no se cuantiza nunca: es un operador de
esperanza integral fijo. Sobrescríbelo con `keep_high_precision=("head.",)`
cuando tengas una razón para hacerlo.

## Los datos de calibración no son datos de entrenamiento

`calib=` toma unos cientos de imágenes, no lee ninguna etiqueta y se ejecuta solo
hacia delante para estimar los rangos de activación. `data=` en `train()` y
`val()` es el dataset etiquetado que se usa para los gradientes y las métricas.
Son argumentos distintos con propósitos distintos, y el valor por defecto de
`calib` es `coco128.yaml`.

`algorithm="minmax"` conserva los extremos absolutos vistos a lo largo de los
batches de calibración y es lo que selecciona `"auto"`. `"percentile"` usa la
media de los percentiles 0,1 y 99,9 de cada batch; se midió que hunde la
precisión de la familia DETR, porque los valores atípicos de las activaciones de
un transformer son estructurales. Lo que de verdad corrige la sensibilidad al
INT8 de los modelos pequeños es calibrar sobre suficientes batches: con el
`coco128` por defecto, YOLO9-t se queda a cerca de un punto de mAP de su
puntuación en float. El algoritmo elegido queda registrado en el manifiesto del
checkpoint.

## Recuperar la precisión

<code-tabs name="train" />

Los módulos cuantizados mantienen pesos maestros en fp32 y aplican cuantización
simulada con un straight-through estimator, así que los gradientes llegan a los
maestros y los entrenadores existentes funcionan sin cambios: EMA, AMP, la
reanudación desde checkpoint y los argumentos de destilación se combinan todos.

QAT es un fine-tuning de un modelo ya entrenado. Usa learning rates de
fine-tuning en lugar de los valores por defecto para entrenar desde cero, o una
ejecución corta destruirá los pesos preentrenados al margen de la cuantización.
La disponibilidad de QAD sigue al soporte de destilación de cada familia, lo que
hoy significa `yolo9` y `rfdetr`.

Los modelos cuantizados con `fp16` y `bf16` son solo de inferencia, y el
entrenador los rechaza remitiendo a `amp=True`.

## Exportación

<code-tabs name="export" />

`format="pt"` cristaliza el modelo. Los pesos y las escalas empaquetados en baja
precisión sustituyen a los maestros, y el resto no cuantizado se convierte a fp16
salvo que se pase `remainder="fp32"`. El invariante del empaquetado es que
desempaquetar reproduce la simulación bit a bit en el dispositivo en el que
finalizaste, así que el archivo final puntúa exactamente lo que validaste.
Medido: YOLO9-s int8 pasa de 29,5 MB a 9,6 MB, y RF-DETR-n nvfp4 de 122 MB a
26 MB. Cargar uno de ellos da un modelo listo para inferencia, y llamar a
`train()` sobre él reconstruye los maestros a partir de los pesos empaquetados
automáticamente.

`format="onnx"` se aplica a los modelos `int8` y emite un grafo QDQ que lleva las
escalas calibradas o entrenadas con QAT del propio modelo, que ONNX Runtime y
TensorRT ejecutan con kernels INT8 reales. Es una ruta distinta de
[`export(format="onnx", int8=True)`](/docs/export/onnx) sobre un modelo float,
donde ONNX Runtime deriva las escalas por su cuenta.

Las recetas de cast no necesitan ningún exportador cuantizado:

<code-tabs name="dequantize" />

## Restricciones

La aritmética cuantizada se ejecuta en simulación, es decir, cuantización
simulada calculada en islas de float32 incluso bajo AMP. La simulación es fiel en
lo numérico, así que una puntuación de `val()` en cualquier dispositivo es una
afirmación real sobre la aritmética cuantizada. No es una afirmación sobre la
velocidad.

Dos excepciones se ejecutan de forma nativa. `fp16` y `bf16` son casts
corrientes. Los módulos `fp8` finalizados ejecutan su GEMM directamente sobre
pesos E4M3 empaquetados a través de `torch._scaled_mm` en hardware de clase Ada,
Hopper y Blackwell, usando las mismas escalas de activación calibradas que la
simulación; poner `LIBREYOLO_KERNELS=off` restaura en todas partes la ruta
simulada exacta.

La cobertura de despliegue es más estrecha que la lista de recetas. Aquí solo
`int8` tiene una forma ONNX desplegable; `fp8` y las recetas lineales por debajo
de 8 bits se ejecutan en PyTorch y cristalizan a través de `format="pt"`.
Pedirles una exportación a ONNX lanza un error con esa indicación, igual que
pedir cualquier formato que no sea ONNX a un modelo `int8`: construye los motores
posteriores a partir del grafo QDQ.

Exportar un modelo `int8` cuyas activaciones nunca se calibraron registra un
aviso y produce un grafo que lleva únicamente la cuantización de los pesos.
