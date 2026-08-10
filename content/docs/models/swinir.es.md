---
title: SwinIR
families: [swinir]
seo_title: "SwinIR: superresolución de imágenes a 4x en LibreYOLO"
description: "Usa SwinIR en LibreYOLO para superresolución de imágenes a 4x. Instala, predice, valida y exporta los checkpoints lightweight, medium y large."
lead: "Una red Swin Transformer para restauración de imágenes. LibreYOLO incluye inferencia y validación para sus checkpoints de superresolución a 4x: el generador oficial lightweight y los generadores real-world medium y large."
keywords: [SwinIR, Swin Transformer, "superresolución de imágenes", "aumentar la resolución de una imagen", "restauración de imágenes python", "escalar imagen 4x sin perder calidad", "residual Swin Transformer block"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSwinIRm-restore.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Por tiles, para imágenes grandes
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile divide el forward pass en tiles solapados y funde de nuevo
        # las costuras; tile_pad es el halo que se añade alrededor de cada
        # tile antes de recortarlo de vuelta. Ambos son argumentos con
        # nombre solo de Python, no flags de la CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # si se omite, imgsz toma por defecto un tamaño de patch interno
        # pequeño, no tu resolución de trabajo, así que pasa el tamaño que
        # tu despliegue le da realmente al modelo.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve
        # el mismo objeto Results.
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
---

## Instalación

SwinIR no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

Un resultado de restauración no lleva cajas; `result.restored` es una imagen
RGB densa `(H, W, 3)` uint8, sobre un lienzo 4x el de entrada en cada
dimensión. `save=True` escribe esa imagen directamente en lugar de un gráfico
anotado. La entrada se rellena hasta un múltiplo de 8 en vez de redimensionarse,
así que la predicción se ejecuta a la resolución propia de la foto; una fuente
más grande de lo que permite la memoria se puede dividir con `tile` y
`tile_pad`, que funden de nuevo las costuras de los tiles en la salida. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Tres tamaños, todos fijos en un escalado 4x. `s` es el generador oficial
lightweight, con cuatro etapas de bloque residual Swin Transformer (RSTB) y
upsampling pixel-shuffle directo. `m` y `l` son los generadores real-world
medium y large, con seis y nueve etapas RSTB y un upsampler de vecino más
cercano más convolución construido para degradaciones del mundo real y no solo
para la reducción bicúbica.

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
checkpoint y devuelve el mismo `Results`. ExecuTorch y todos los formatos que
la matriz marca como bloqueados no están disponibles para esta familia; ONNX,
TorchScript, TensorRT, OpenVINO y TFLite sí lo están. [Exportación](/docs/export)
enumera los argumentos que acepta cada formato y los extras que añaden unos
pocos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
