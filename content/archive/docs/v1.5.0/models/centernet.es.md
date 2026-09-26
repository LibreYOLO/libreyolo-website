---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: detección de objetos en LibreYOLO'
description: >-
  Ejecuta CenterNet (Objects as Points) en LibreYOLO con los backbones ResDCN-18
  y DLA-34. Predice, valida y exporta a ONNX bajo licencia MIT. Sin ruta de
  entrenamiento.
lead: >-
  CenterNet modela un objeto como el punto central de su bounding box y regresa
  todas las demás propiedades a partir de un pico del heatmap, así que no
  necesita anchors ni un paso de non-maximum-suppression. LibreYOLO lo incluye
  como detector solo de inferencia.
keywords:
  - CenterNet
  - Objects as Points
  - detección de objetos python
  - detector anchor-free
  - detección por keypoints
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # La exportación a ONNX necesita opset 16 o superior: la etapa de
        # upsampling por convolución deformable baja a GridSample, que se
        # introdujo en el opset 16.
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve
        # el mismo objeto Results.
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Instalación

CenterNet no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran los picos del heatmap ya ordenados; `iou` se acepta por paridad de API
pero no tiene efecto, porque el decodificado top-k de picos de CenterNet no
necesita ningún paso de supresión por IoU de cajas. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Dos backbones. `resdcn18` combina un tronco ResNet-18 con upsampling por
convolución deformable; `dla34` combina un tronco DLA-34 con upsampling por
agregación profunda iterativa. Ambos alimentan las mismas tres cabezas densas
(heatmap, ancho/alto, offset) y el mismo lienzo de entrada.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

La exportación a ONNX requiere opset 16 o superior: la etapa de upsampling por
convolución deformable de ambos backbones baja al operador `GridSample` de
ONNX, que se introdujo en el opset 16. Pedir un opset anterior lanza un error
antes de que empiece el trazado.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

El grafo de ResDCN-18 acredita además el human-pose-estimation.pytorch de
Microsoft, con licencia MIT, y el grafo de DLA-34 acredita la implementación de
DLA de Fisher Yu, con licencia BSD-3-Clause. LibreYOLO no incorpora la extensión
DCNv2 original que usaba el proyecto upstream; la ejecución nativa usa en su
lugar `deform_conv2d` de torchvision, con licencia BSD-3-Clause, y la
implementación portable, solo para exportación, se escribió aparte para
LibreYOLO.

</provenance-box>

## Cita

<citation-block />
