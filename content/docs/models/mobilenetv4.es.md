---
title: MobileNetV4
families: [mobilenetv4]
seo_title: "MobileNetV4: entrena, valida y exporta bajo Apache-2.0"
description: "Usa MobileNetV4 en LibreYOLO para clasificación de imágenes. Instala, predice, haz fine-tuning, valida y exporta LibreMobileNetV4 small/medium/large."
lead: "MobileNetV4 es un clasificador de imágenes pensado para hardware móvil y edge, que usa el bloque Universal Inverted Bottleneck para unificar varios diseños de bloques móviles previos en una única estructura sobre la que buscar. LibreYOLO lo soporta para una tarea: clasificación."
keywords: [MobileNetV4, "MobileNetV4 conv", "clasificación de imágenes python", "inferencia en móvil", "clasificador edge", "clasificador ImageNet"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMobileNetV4s-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMobileNetV4s-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMobileNetV4s-cls.pt format=onnx
        libreyolo export model=LibreMobileNetV4s-cls.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreMobileNetV4s-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Instalación

MobileNetV4 no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro modelo es un cambio de una línea. Un clasificador no lleva
cajas ni máscaras: `result.probs` contiene la predicción de la imagen completa,
con `top1`, `top5`, `top1conf` y `top5conf`. `conf`, `iou` y `max_det` se
aceptan por paridad de API, pero no tienen efecto, porque no hay nada que
umbralizar ni suprimir en un único vector de probabilidades. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Tres tamaños, small/medium/large, todos solo convolucionales: esta familia
excluye las variantes híbridas que añaden atención Mobile MQA. Elegir un tamaño
es un intercambio directo entre número de parámetros y precisión. La tarea es
fija: todos los tamaños cubren solo clasificación. El nombre del archivo de
pesos termina en `-cls.pt` en todos los tamaños, y ese sufijo es lo que lee la
factoría para enrutar a esta familia; no hace falta ningún argumento `task=`.

## Entrenamiento

El fine-tuning parte del backbone de ImageNet publicado y reconstruye
automáticamente la capa final del clasificador para que coincida con el número
de clases del dataset objetivo.

<code-tabs name="train" />

Si no tocas nada, el trainer ejecuta 100 épocas con `lr0=1e-3` y AdamW, un
batch de 64 y parada temprana tras 50 épocas sin mejora. `data` acepta la raíz
de un dataset (`train/` y `val/`, una carpeta por clase), un nombre corto
conocido como `imagenette160`, o una URL a un `.zip`. Aquí no se soporta
`lora=True`; pasarlo lanza un error, porque LoRA en LibreYOLO actúa sobre
componentes de tipo transformer con capas `nn.Linear`, y los bloques UIB de
esta familia no tienen ninguna.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/`. En clasificación son la
precisión top-1 y top-5 sobre el split de validación.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) lista los
argumentos que acepta cada formato y los extras que añaden algunos de ellos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>
