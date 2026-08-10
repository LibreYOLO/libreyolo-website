---
title: MoGe-2
families: [moge2]
seo_title: "MoGe-2: predice, valida y exporta normales de superficie"
description: "Usa MoGe-2 en LibreYOLO para predicción densa de normales de superficie. Instala, predice, valida y exporta los checkpoints oficiales ViT-S, ViT-B y ViT-L."
lead: "MoGe-2 es un modelo de geometría monocular de una sola pasada que predice un campo denso de normales de superficie a partir de una única imagen RGB. LibreYOLO lo soporta solo para estimación de normales, a través de los checkpoints oficiales ViT-S, ViT-B y ViT-L."
keywords: [MoGe-2, MoGe 2, "estimación de normales de superficie", "mapa de normales", "geometría monocular", "predicción densa python", DINOv2]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # vectores unitarios float32 (H, W, 3)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMoGe2s-normal.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # grados
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # porcentaje de píxeles
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518
        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518 half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
---

## Instalación

MoGe-2 no necesita ningún extra opcional. Todo lo que importa está en la instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan automáticamente en el primer uso: LibreYOLO obtiene el
tamaño correspondiente directamente de los checkpoints oficiales y lo guarda en
la caché local.

<code-tabs name="predict" />

MoGe-2 devuelve un campo denso en lugar de un conjunto de detecciones, así que
`result.boxes` está vacío y `conf`, `iou` y `max_det` no tienen efecto.
`result.normal_map` contiene el resultado: un array `(H, W, 3)` de vectores
unitarios en el sistema de coordenadas de cámara de OpenCV, donde `+x` es hacia
la derecha, `+y` hacia abajo, `+z` hacia el interior de la escena, y una
superficie orientada hacia la cámara se lee como `(0, 0, -1)`. Predecir una
lista de imágenes ejecuta una pasada hacia delante por imagen; esta familia no
tiene una ruta rápida de batch apilado. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Tres tamaños de encoder se publican como checkpoints separados: ViT-S, ViT-B y
ViT-L, todos con la misma resolución de entrada. El banco de pruebas de
LibreYOLO no ha medido esta familia, así que no hay cifras de precisión
publicadas con las que compararlos; elige un tamaño según tu propio presupuesto
de cómputo.

## Validación

`val()` mide el error angular contra un dataset emparejado de mapas de
normales: imágenes junto a PNG de normales de 16 bits con el mismo nombre base,
con una máscara de validez opcional para que los píxeles rellenados e inválidos
nunca cuenten. Devuelve el error angular medio y mediano en grados, más el
porcentaje de píxeles dentro de 11,25, 22,5 y 30 grados.

<code-tabs name="val" />

## Exportación

<export-matrix />

La exportación de normales usa un contrato de ejecución de resolución fija y
batch 1: se rechazan `dynamic` y cualquier `batch` distinto de 1, y `imgsz` debe
ser divisible por el tamaño de parche del encoder ViT, algo que LibreYOLO
comprueba antes de que empiece la ejecución. Un artefacto exportado se vuelve a
cargar con `LibreYOLO()` según su extensión de archivo, así que un archivo
`.onnx` se comporta como un checkpoint y devuelve el mismo `Results`.

<code-tabs name="export" />

## Licencia

<provenance-box>

LibreYOLO no copia estos checkpoints a su propia organización.
`LibreYOLO("LibreMoGe2s-normal.pt")` descarga el tamaño correspondiente
directamente de los repositorios oficiales de Hugging Face en una revisión
fijada, y verifica el archivo contra un checksum SHA-256 registrado antes de
usarlo.

</provenance-box>

## Cita

<citation-block />
