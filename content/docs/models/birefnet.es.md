---
title: BiRefNet
families: [birefnet]
seo_title: "BiRefNet: eliminación de fondo y matting en LibreYOLO"
description: "Usa BiRefNet en LibreYOLO para eliminar el fondo y para segmentación dicotómica de imágenes. Instala, predice, valida y exporta el checkpoint general."
lead: "Una red de referencia bilateral que predice un alpha matte suave que separa al sujeto de su fondo. LibreYOLO incluye inferencia y validación para la tarea de matte de BiRefNet."
keywords: [BiRefNet, "quitar el fondo de una imagen", "eliminar fondo python", "segmentación dicotómica de imágenes", "alpha matte", "image matting", "recorte con transparencia"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreBiRefNetl-matte.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Recorte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: el RGB de origen más el matte como canal alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Un directorio que contenga images/ y un directorio de mattes
        # autodetectado (mattes/, matte/, gt/, masks/, mask/ o alpha/)
        # también vale en lugar de un YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve
        # el mismo objeto Results.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
---

## Instalación

BiRefNet no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

Un resultado de matte no lleva cajas; `result.matte` es un array denso
`(H, W)` float32 en `[0, 1]`, donde 1 es primer plano puro y 0 fondo puro. A
diferencia de una máscara binaria, el matte suave conserva el detalle de los
bordes con antialiasing, como el pelo o el pelaje. `result.cutout()` compone
la imagen de origen con ese canal alfa en un array RGBA, y `result.save(path)`
(o `save=True` en la llamada de predicción) lo escribe directamente a un PNG
con fondo transparente. El modelo funciona sobre un lienzo nativo fijo de
1024x1024; no se admite otra resolución, porque las tablas de posición
relativa del backbone Swin están atadas a ella, y un desajuste las interpola
mal en lugar de lanzar un error. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Un único checkpoint publicado, `l`, el modelo BiRefNet-general del nivel
Swin-L y la opción por defecto de calidad en el proyecto original. El código
de la familia también soporta un nivel lite Swin-T, `t`, pero todavía no hay
publicada ninguna conversión suya a LibreYOLO.

## Validación

`val()` informa de dos métricas sobre una carpeta emparejada de imágenes y
mattes, ambas en `[0, 1]` e independientes de la resolución: MAE, el error
absoluto medio frente al alfa del ground truth (mejor cuanto más bajo), y
S-measure (Fan et al., ICCV 2017), una similitud estructural que premia
conservar la forma y los huecos del sujeto, algo que el MAE por píxel por sí
solo se pierde (mejor cuanto más alto). La validación pasa por el propio
`predict` del modelo, así que usa exactamente el preprocesado de la familia.

<code-tabs name="val" />

La validación es solo de inferencia; el fine-tuning es una continuación
documentada, no una función ya incluida (consulta Predicción para la
restricción exacta de resolución que heredaría cualquier trainer futuro).

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` se comporta como un checkpoint y
devuelve el mismo `Results`. TorchScript es el camino validado; la conversión
a ONNX funciona, pero no ha superado el mismo listón de paridad.
[Exportación](/docs/export) enumera los argumentos que acepta cada formato y
los extras que añaden unos pocos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
