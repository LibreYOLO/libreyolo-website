---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: segmentación semántica en LibreYOLO'
description: >-
  Usa SegFormer en LibreYOLO para segmentación semántica sobre ADE20K en los
  tamaños b0-b5. Instala, predice, entrena y exporta; los pesos preentrenados
  son de uso no comercial.
lead: >-
  SegFormer es un transformer de segmentación semántica que combina un encoder
  jerárquico Mix Transformer (MiT) con una cabeza de decodificación all-MLP
  ligera, evitando los decoders pesados y las codificaciones posicionales fijas
  que necesitaban los transformers de segmentación anteriores. LibreYOLO lo
  soporta para una sola tarea, segmentación semántica, en seis tamaños.
keywords:
  - SegFormer
  - segmentación semántica python
  - Mix Transformer
  - MiT
  - transformer de segmentación
  - ADE20K
  - segmentar imágenes por píxel
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (fine-tuning)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Desde cero
      language: python
      code: >
        from libreyolo.models.segformer.model import LibreSegformer


        # Sin model_path: inicialización aleatoria, no se descarga nada. La
        única

        # vía a pesos libres del término no comercial de los checkpoints

        # preentrenados.

        model = LibreSegformer(size="b0", nb_classes=150)

        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Instalación

SegFormer no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

`result.semantic_mask` lleva el mapa denso de clases: `.data` es un tensor
`(H, W)` de ids de clase al tamaño original de la imagen, y `.classes` enumera
los ids de clase realmente presentes. `result.boxes` es `None`, ya que no hay
detecciones por instancia. `conf` e `iou` se aceptan por paridad de API pero no
cambian la salida: el modelo devuelve una clase por píxel, no detecciones por
instancia que filtrar o deduplicar. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Seis tamaños, de b0 a b5, que ensanchan y profundizan el encoder Mix Transformer
en cada paso manteniendo el mismo diseño de cabeza de decodificación all-MLP.

<checkpoint-table />

## Entrenamiento

`train()` hace fine-tuning de un checkpoint publicado por defecto. Si en su
lugar no le pasas ningún `model_path` a `LibreSegformer(...)`, el modelo se
construye con el encoder y la cabeza inicializados al azar y entrena desde cero,
la única vía a pesos que no arrastran ninguna restricción no comercial de los
checkpoints preentrenados (consulta [Licencia](#licensing)).

<code-tabs name="train" />

Si lo dejas como está, el trainer sigue la receta de ADE20K del paper de
SegFormer: AdamW con un learning rate base para el backbone y la cabeza de
decodificación entrenada a 10x esa tasa, weight decay en todas partes excepto en
LayerNorm y en la convolución posicional del Mix-FFN, y un schedule de
decaimiento lineal con warmup. La convergencia de los tamaños más grandes, de b3
a b5, no se ha validado de principio a fin.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos, multi-GPU
y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/`: mIoU y precisión por
píxel, medidos contra cualquier dataset en el formato con el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

El encoder y la cabeza de decodificación de LibreSegformer son un port a PyTorch
de la implementación de SegFormer de Hugging Face Transformers, con licencia
Apache-2.0, y no de NVlabs/SegFormer: el repositorio original de NVIDIA nunca se
leyó ni se copió, y se cita aquí solo para atribuir el trabajo a los autores del
paper. Solo los checkpoints preentrenados de arriba llevan la restricción no
comercial de NVIDIA; la arquitectura y el propio código de LibreYOLO siguen
siendo MIT en todo momento.

</provenance-box>

## Cita

<citation-block />
