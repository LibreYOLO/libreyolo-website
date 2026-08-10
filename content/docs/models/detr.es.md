---
title: DETR
families:
  - detr
seo_title: 'DETR: predecir y exportar bajo Apache-2.0'
description: >-
  Ejecuta DETR, el transformer de detección original, en LibreYOLO. Instala,
  predice, valida y exporta cuatro tamaños basados en ResNet, todos con licencia
  Apache-2.0.
lead: >-
  DETR es el transformer de detección original: predice un conjunto fijo de
  objetos con un decoder transformer emparejado por el algoritmo húngaro, en
  lugar de usar anchors o una rejilla densa. LibreYOLO incluye cuatro tamaños
  para detección, solo de inferencia.
keywords:
  - DETR
  - detection transformer
  - detección de objetos python
  - transformer de detección
  - emparejamiento húngaro
  - DETR pytorch
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() devuelve un dict plano, no un objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        Results.

        model = LibreYOLO("LibreDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Instalación

DETR no necesita ningún extra opcional. Todo lo que importa está en la
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
filtran la selección de queries; `iou` se acepta por paridad de API pero no
tiene efecto, porque el decoder predice un conjunto completo y no tiene paso de
NMS. Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

En LibreYOLO, DETR es solo de inferencia. El proyecto upstream entrena durante
500 épocas con emparejamiento húngaro; esa receta no está implementada aquí,
así que `train()` lanza `NotImplementedError`.

## Variantes

Cuatro checkpoints combinan dos profundidades de backbone, ResNet-50 o
ResNet-101, con una etapa C5 dilatada opcional: las variantes DC5 mantienen la
última etapa del backbone a resolución completa en lugar de reducirla más, así
que el decoder lee un mapa de características más fino a partir del mismo
tamaño de entrada. Las cuatro comparten 100 object queries aprendidas y un
encoder-decoder transformer de seis capas, y todas se ejecutan a la misma
resolución de entrada.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según la extensión
de su archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>
