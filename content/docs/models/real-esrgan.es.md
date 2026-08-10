---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: superresolución de imágenes en LibreYOLO'
description: >-
  Usa Real-ESRGAN en LibreYOLO para superresolución de imágenes práctica a 4x,
  2x y un nivel 4x rápido. Instala, predice, valida y exporta.
lead: >-
  Un escalador de superresolución ciega pensado para uso real, entrenado con
  degradaciones sintéticas en lugar de solo con reducción bicúbica. LibreYOLO
  incluye inferencia y validación para sus checkpoints 4x, 2x y 4x rápido.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - superresolución de imágenes
  - mejorar la calidad de una imagen python
  - aumentar la resolución de una imagen
  - escalar imagen sin perder calidad
  - restauración de imágenes
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 'Por tiles, para imágenes grandes'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile divide el forward pass en tiles solapados y funde de nuevo
        # las costuras; tile_pad es el halo que se añade alrededor de cada
        # tile antes de recortarlo de vuelta. Ambos son argumentos con
        # nombre solo de Python, no flags de la CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # si se omite, imgsz toma por defecto un tamaño de patch interno
        # pequeño, no tu resolución de trabajo, así que pasa el tamaño que
        # tu despliegue le da realmente al modelo.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve
        # el mismo objeto Results.
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Instalación

Real-ESRGAN no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

Un resultado de restauración no lleva cajas; `result.restored` es una imagen
RGB densa `(H, W, 3)` uint8, sobre un lienzo `Results.restore_scale` veces el
de entrada en cada dimensión. `save=True` escribe esa imagen directamente en
lugar de un gráfico anotado. La entrada se convierte a RGB y se descarta
cualquier canal alfa. Una fuente más grande de lo que permite la memoria se
puede dividir con `tile` y `tile_pad`, que funden de nuevo las costuras de los
tiles en la salida. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

Tres checkpoints, nombrados por su factor de escalado. `x4` es RRDBNet
(`RealESRGAN_x4plus`), 23 bloques densos residual-in-residual, la opción por
defecto de calidad a 4x. `x2` es la misma arquitectura RRDBNet a 2x. `x4t` es
SRVGGNetCompact (`realesr-general-x4v3`), un generador más pequeño y rápido
construido para vídeo y usos de menor latencia a 4x. El modelo general del
proyecto original incluye además una red emparejada de fuerza de denoise que se
mezcla en el momento de la inferencia; ese control de fuerza no forma parte de
este port, que ejecuta el generador `x4t` base.

## Validación

`val()` mide PSNR y SSIM entre la salida restaurada y una imagen objetivo
limpia, ambas calculadas en RGB sobre el lienzo original, sin recorte de bordes
y sin redimensionar. SSIM usa una ventana gaussiana de 11x11 con sigma 1.5,
promediada sobre los tres canales de color.

<code-tabs name="val" />

El argumento de dataset es un YAML que empareja un directorio de imágenes de
entrada degradadas con un directorio de imágenes objetivo limpias de la misma
resolución; consulta [formatos de dataset](/docs/reference/dataset-formats)
para las claves exactas.

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato y los extras que añaden unos pocos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
