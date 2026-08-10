---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: quitar ruido, entrenar y exportar con licencia MIT'
description: >-
  Usa NAFNet en LibreYOLO para eliminar ruido y restaurar imágenes. Instala,
  predice, entrena, valida y exporta el checkpoint SIDD, con licencia MIT.
lead: >-
  NAFNet es una red convolucional para restauración de imágenes que elimina las
  funciones de activación no lineales del bloque UNet típico y las sustituye por
  una multiplicación elemento a elemento. LibreYOLO lo soporta para una tarea,
  la restauración, con un checkpoint publicado de eliminación de ruido en
  imágenes reales entrenado sobre SIDD.
keywords:
  - NAFNet
  - restauración de imágenes
  - quitar ruido de una imagen
  - eliminar ruido de una foto python
  - eliminar desenfoque de una imagen
  - denoising de imágenes
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Guardar la imagen restaurada
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Procedencia del checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation y dataset se registran en el checkpoint guardado; no
        # cambian lo que se entrena.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() devuelve un dict simple, no un objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        Results.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model("noisy.jpg")


        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Instalación

NAFNet no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto lleva un único campo para esta familia,
`restored`, una imagen RGB uint8 densa en formato HWC sobre el lienzo
original; no hay boxes que recorrer. `save=True` escribe esa imagen restaurada
directamente en disco en lugar de dibujar una anotación sobre la entrada.
`conf`, `iou` y `max_det` se aceptan por paridad de firma con el resto de
familias, pero no tienen efecto, ya que la restauración no produce
detecciones que filtrar. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

Dos anchos comparten esta arquitectura: `s` (ancho 32) y `l` (ancho 64),
ambos construidos en torno a un parche de entrenamiento de 256 px. La
predicción y la validación se ejecutan a la resolución nativa de la imagen sea
cual sea su tamaño, con padding solo hasta el factor de submuestreo de la red.
De momento solo está publicado el ancho `l`, como checkpoint de eliminación de
ruido en imágenes reales entrenado sobre SIDD.

## Entrenamiento

NAFNet hace fine-tuning sobre tus propias parejas de imágenes
degradada/limpia: un YAML de dataset que apunte a una carpeta `inputs/<split>/`
con las imágenes degradadas y a una carpeta `targets/<split>/` con los
objetivos limpios, emparejadas por el nombre de archivo sin extensión.
`degradation` y `dataset` son cadenas opcionales que se registran en el
checkpoint guardado como procedencia; no intervienen en el entrenamiento.

<code-tabs name="train" />

Si no lo tocas, el trainer ejecuta 100 epochs con AdamW a `lr0=1e-3`, un batch
de 16, recortes de 256 px y parada temprana tras 50 epochs sin mejora de PSNR.
Esta familia no tiene camino LoRA: `lora=True` lanza un error en lugar de
ejecutarse, porque `NAFNetTrainer` nunca se acoge al fine-tuning con
adaptadores.

Durante el entrenamiento la red funciona con global-average pooling normal. El
pooling local por ventanas de NAFNet, exclusivo de inferencia (Test-time Local
Converter), se desconecta antes del primer epoch y se vuelve a conectar cuando
termina el entrenamiento, ya que retropropagar a través de un pooling local de
ventana fija no se correspondería con la forma en que se usa el checkpoint en
inferencia.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos,
multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario con `metrics/PSNR` y `metrics/SSIM`,
calculados en RGB sobre todo el lienzo válido: SSIM usa una ventana gaussiana
de 11x11 con sigma 1.5, y el `fitness` con el que se elige el mejor checkpoint
es el valor de PSNR. `data` apunta al mismo formato de dataset de imágenes
emparejadas que se usa para entrenar.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`, con `restored` llevando la imagen de
salida. NAFNet se exporta a una resolución espacial fija: `imgsz` debe ser
divisible por el factor de submuestreo de la red (16 para ambos anchos de la
arquitectura), y solo la dimensión de batch es dinámica cuando `dynamic=True`;
el alto y el ancho quedan fijados en el momento de la exportación.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
