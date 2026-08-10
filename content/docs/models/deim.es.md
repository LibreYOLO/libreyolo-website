---
title: DEIM
families: [deim]
seo_title: "DEIM y DEIMv2 en LibreYOLO"
description: "Usa DEIM y DEIMv2 en LibreYOLO para detección de objetos. Instala, predice, entrena, valida y exporta, desde un tamaño de medio millón de parámetros en adelante."
lead: "Un transformer de detección entrenado con emparejamiento denso uno a uno, que converge en muchas menos épocas que las recetas DETR sobre las que se construye. LibreYOLO incluye dos versiones, que se distinguen por el checkpoint que cargues."
keywords: [DEIM, DEIMv2, DINOv3, "transformer de detección", DETR, "detección de objetos", "detección de objetos en tiempo real", "detección de objetos python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La versión forma parte del nombre del archivo, y la factoría enruta
        # según el checkpoint, así que ambas se cargan igual.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Cualquier fuente que acepta la biblioteca: archivo, carpeta, URL,
        # índice de webcam, stream RTSP o una lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml descarga una muestra de 128 imágenes en el primer uso.
        # Apunta `data` al YAML de tu propio dataset para una ejecución real.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Si no se indican, epochs, batch, imgsz y lr0 salen de la receta
        # publicada para el tamaño que se haya cargado.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Necesita el extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() devuelve un dict plano, no un objeto
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Contra COCO
      language: bash
      code: |
        # coco-val-only.yaml descarga las 5000 imágenes de val2017 y se salta
        # el conjunto de entrenamiento. Lleva un script de descarga embebido,
        # así que necesita permiso explícito salvo que el dataset ya esté en
        # local.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Necesita el extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve el
        # mismo objeto Results.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

Ninguna de las dos versiones necesita un extra opcional. Todo lo que importan
está en la instalación base.

```bash
pip install libreyolo
```

El fine-tuning con adaptadores mediante `lora=True` es la excepción, y necesita
el extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran un decodificado top-k sobre queries y clases; no hay un paso de NMS que
ajustar, e `iou` se acepta pero no se usa. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

La versión 1 trae cinco tamaños, todos con el mismo tamaño de entrada. La
versión 2 conserva esos cinco nombres y añade tres más pequeños, `atto`, `femto`
y `pico`, los dos primeros nativos a un tamaño de entrada menor que el resto.
Por tanto, cinco códigos de tamaño existen en ambas versiones y nombran modelos
distintos; la versión va escrita en el nombre del archivo del checkpoint.

<benchmark-table task="detect" />

<va-embed />

La versión 1 mantiene la arquitectura de D-FINE y cambia su objetivo de
clasificación por la función de pérdida sensible a la emparejabilidad de la
receta densa uno a uno, así que las dos familias comparten casi todas las claves
del state dict y se distinguen por los metadatos del checkpoint. La versión 2
mantiene ese contrato de entrenamiento y mezcla backbones: HGNetv2 por debajo de
`s`, y un vision transformer DINOv3 con un adaptador de ajuste espacial en `s` y
por encima. Ese backbone es lo que pone una segunda licencia sobre esos cuatro
checkpoints, así que lee [licencia](#licensing) antes de llevar uno a
producción.

## Entrenamiento

El entrenamiento parte de un checkpoint publicado. `pretrained` nunca llega al
trainer: la versión 1 avisa de que la clave es desconocida y la ignora, la
versión 2 la elimina. Ninguna de las dos te da un modelo inicializado
aleatoriamente.

<code-tabs name="train" />

Pasa `lr0` tú mismo en la versión 1. Su firma de `train()` en Python usa `4e-4`
por defecto, el valor de la receta publicada de COCO, mientras que la
configuración de entrenamiento de la familia lleva `1e-4` como valor por defecto
para el fine-tuning, y ese valor más bajo es el que resuelve la CLI cuando falta
el argumento. La configuración deja constancia de la medición que hay detrás:
con los tamaños de batch que usa un fine-tuning real, sobre datasets pequeños,
el learning rate de COCO degradaba la transferencia de forma medible.

La versión 2 resuelve esos valores por defecto por su cuenta. Si dejas `epochs`,
`batch`, `imgsz` y `lr0` sin indicar, lee cada uno de la receta publicada para
el tamaño que se haya cargado, así que los tamaños pequeños entrenan a su propia
resolución de entrada sin que haya que decírselo, y un valor que pases tú tiene
prioridad sobre la receta. `imgsz` es el argumento que sí restringe: tiene que
ser un múltiplo positivo de 32, y si no, la versión 2 lanza un error antes de
que empiece la ejecución.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

Las filas de la tabla de benchmarks de arriba salen del arnés de benchmarks de
LibreYOLO; la nota bajo esa tabla indica qué dataset las produjo y enlaza los
registros de las ejecuciones.

## Exportación

<export-matrix />

La matriz cubre las dos versiones en una sola página: donde discrepan sobre un
formato, la celda muestra la más débil de las dos, así que aquí nada queda
sobrevendido para la versión que cargues.

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>
Los cuatro tamaños de DEIMv2 de S en adelante toman su backbone de DINOv3, así
que sus repositorios de pesos llevan tanto Apache-2.0 como la DINOv3 License de
Meta, y LibreYOLO distribuye el código del backbone DINOv3 bajo ese mismo
acuerdo. El resto de esta familia, incluidos todos los tamaños de DEIMv2 por
debajo de S, es solo Apache-2.0.
</provenance-box>

## Cita

<citation-block />

DEIMv2 es un artículo aparte y tiene su propio bloque de cita en
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
cita ese si has usado un checkpoint de la versión 2.
