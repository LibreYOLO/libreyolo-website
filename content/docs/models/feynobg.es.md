---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: eliminación de fondo en LibreYOLO'
description: >-
  Usa FeyNobg en LibreYOLO para eliminar el fondo y para alpha matting, una
  variante de BiRefNet más profunda creada por Feyn Inc. Instala, predice y
  valida.
lead: >-
  Un modelo de eliminación de fondo de Feyn Inc. que profundiza la arquitectura
  de BiRefNet y la reentrena. LibreYOLO incluye inferencia y validación para la
  tarea de matte de FeyNobg.
keywords:
  - FeyNobg
  - quitar el fondo de una imagen
  - eliminar fondo python
  - segmentación dicotómica de imágenes
  - alpha matte
  - image matting
  - recorte con transparencia
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Recorte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: el RGB de origen más el matte como canal alfa.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # Un directorio que contenga images/ y un directorio de mattes
        # autodetectado (mattes/, matte/, gt/, masks/, mask/ o alpha/)
        # también vale en lugar de un YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Instalación

FeyNobg no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

El checkpoint se descarga de la organización LibreYOLO en Hugging Face en el
primer uso y se guarda en la caché local, igual que en cualquier otra familia,
aunque todavía no aparece en la tabla de Checkpoints de esta página.

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

Un único tamaño publicado, `l`, un backbone del nivel Swin-L. FeyNobg toma la
arquitectura de BiRefNet y profundiza su tercera etapa Swin de 18 a 24 bloques
antes de reentrenarla, así que el port a LibreYOLO reutiliza el forward pass,
el preprocesado y el contrato de salida de un único logit de BiRefNet; la
predicción, la validación y el manejo de checkpoints se comportan igual que en
la familia `birefnet`.

## Validación

`val()` informa de dos métricas sobre una carpeta emparejada de imágenes y
mattes, ambas en `[0, 1]` e independientes de la resolución: MAE, el error
absoluto medio frente al alfa del ground truth (mejor cuanto más bajo), y
S-measure (Fan et al., ICCV 2017), una similitud estructural que premia
conservar la forma y los huecos del sujeto, algo que el MAE por píxel por sí
solo se pierde (mejor cuanto más alto). La validación pasa por el propio
`predict` del modelo, así que usa exactamente el preprocesado de la familia.

<code-tabs name="val" />

La validación es solo de inferencia. La biblioteca original `nobg` incluye
código de entrenamiento con licencia Apache-2.0; hoy hacer fine-tuning
significa entrenar allí y convertir el resultado con el propio script de
conversión de LibreYOLO, no llamar a `train()` en esta familia, que lanza un
error en lugar de ejecutar un trainer parcial.

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
