---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: predice y exporta, con licencia Apache-2.0'
description: >-
  Usa Deformable DETR en LibreYOLO para detección de objetos. Instala, predice,
  valida y exporta cinco tamaños con atención dispersa, todos con licencia
  Apache-2.0.
lead: >-
  Deformable DETR sustituye la cross-attention densa de DETR por un muestreo
  disperso y multiescala alrededor de cada punto de referencia, que es lo que
  hizo entrenables en la práctica a los detectores transformer. LibreYOLO
  incluye cinco tamaños para detección, solo inferencia.
keywords:
  - Deformable DETR
  - transformer de detección
  - atención dispersa
  - atención multiescala
  - detección de objetos
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() devuelve un dict normal, no un objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo

        # objeto Results.

        model = LibreYOLO("LibreDeformableDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Instalación

Deformable DETR no necesita ningún extra opcional. Todo lo que importa está en
la instalación base, con un núcleo de atención deformable multiescala en PyTorch
puro.

```bash
pip install libreyolo
```

Instalar `libreyolo[hub-kernels]` es opcional. Una vez presente el paquete
`kernels`, LibreYOLO descarga en tiempo de ejecución un kernel compilado de
atención deformable multiescala desde el Hugging Face Hub y lo usa en lugar del
núcleo en PyTorch puro; `LIBREYOLO_HUB_KERNELS=0` lo vuelve a desactivar.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran la selección de queries; `iou` se acepta por paridad de API pero no
tiene efecto, porque el decoder es un predictor de conjuntos sin paso de NMS.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

Deformable DETR es solo de inferencia en LibreYOLO. Upstream lo entrena con
matching húngaro y una focal loss de clasificación; esa receta no está
implementada aquí, así que `train()` lanza `NotImplementedError`.

## Variantes

Cinco checkpoints cubren las configuraciones publicadas, todas a la misma
resolución de entrada. `r50ss` restringe la atención a una sola escala de
características; `r50ssdc5` añade encima una etapa C5 dilatada en el backbone.
`r50` es la configuración multiescala por defecto, que muestrea en cuatro
niveles de mapas de características. `r50refine` añade refinamiento iterativo de
los bounding boxes a lo largo de las capas del decoder, y `r50twostage` genera
sus propuestas de región iniciales a partir de la salida del encoder en lugar de
usar queries aprendidas.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`.
[Exportación](/docs/export) lista los argumentos que acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
