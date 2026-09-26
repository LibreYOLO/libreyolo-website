---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: profundidad monocular ligera en LibreYOLO'
description: >-
  Usa ZipDepth en LibreYOLO para estimación de profundidad monocular ligera.
  Instala, predice, valida y exporta dos checkpoints con licencia MIT.
lead: >-
  ZipDepth es una CNN compacta y reparametrizable destilada de Depth Anything V2
  Large que predice un mapa denso de profundidad inversa relativa. LibreYOLO lo
  soporta para la tarea de profundidad: predicción y validación zero-shot, sin
  ruta de entrenamiento.
keywords:
  - ZipDepth
  - estimación de profundidad monocular
  - modelo de profundidad para edge
  - profundidad relativa
  - mapa de profundidad python
  - CNN reparametrizable
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Checkpoint NPU/edge
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El mismo encoder, con una cabeza de upsampling sin unfold para
        # compiladores que no soportan gather/unfold. La salida es visualmente
        # equivalente a la del checkpoint b.
        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibreZipDepthb-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Instalación

ZipDepth no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

`result.depth_map` lleva un mapa denso de profundidad inversa relativa: los
valores más altos significan más cerca de la cámara, y los valores no tienen
unidad métrica ni escala común entre imágenes. `save=True` escribe en disco una
visualización de ese mapa con un mapa de color; `Results.plot()` no cubre esta
familia, ya que está definido solo para normales de superficie y bordes.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Dos checkpoints, ambos con la misma capacidad de encoder, que se diferencian
solo en la cabeza de upsampling entrenada. `b` usa upsampling convexo y se
ejecuta en GPU o CPU. `bnpu` cambia esa cabeza por un decoder sin unfold para
NPU y compiladores edge que no soportan gather/unfold; su salida está
documentada como visualmente equivalente a la de `b`. Elige `bnpu` cuando el
destino de la exportación sea un runtime limitado, y `b` en el resto de casos.

Ambos checkpoints se destilaron a partir de pseudo-etiquetas de Depth Anything
V2 Large, así que esta familia es el nivel compacto y orientado a edge de la
tarea de profundidad de LibreYOLO, junto a los encoders más grandes de Depth
Anything V2.

El entrenamiento no se ofrece para esta familia. `LibreZipDepth.train()` lanza
`NotImplementedError` de forma incondicional: la receta upstream destila
pseudo-etiquetas sobre un conjunto grande de imágenes que no es reproducible
como un entrenamiento de LibreYOLO. Entrena upstream en
[fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) y convierte el
resultado con `weights/convert_zipdepth_weights.py`.

## Validación

`val()` ejecuta el validador de profundidad compartido: alinea cada predicción
con su ground truth mediante una escala y un desplazamiento por mínimos
cuadrados calculados para cada imagen, y después reporta las métricas estándar
de profundidad relativa zero-shot, AbsRel, RMSE y los tres umbrales delta.

<code-tabs name="val" />

## Exportación

<export-matrix />

La exportación sigue un contrato denso de resolución fija: la imagen de origen
se reescala por estiramiento al lienzo exportado, y el mapa de profundidad
devuelto se reescala después de vuelta al lienzo original. Un artefacto
exportado se vuelve a cargar con `LibreYOLO()` según su extensión de archivo,
así que un archivo `.onnx` o `.ncnn` se comporta como un checkpoint y devuelve
el mismo `Results`, con `depth_map` en lugar de boxes.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
