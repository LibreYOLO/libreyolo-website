---
title: Depth Anything V2
families: [depth_anything]
seo_title: "Depth Anything V2: predice y valida profundidad monocular"
description: "Usa Depth Anything V2 en LibreYOLO para estimación de profundidad monocular. Instala, predice y valida; Small se publica con Apache-2.0, y Base y Large con CC-BY-NC-4.0."
lead: "Depth Anything V2 es un encoder DINOv2 acompañado de un decoder DPT que predice un mapa denso de profundidad inversa relativa a partir de una sola imagen. LibreYOLO lo soporta para la tarea de profundidad: predicción y validación zero-shot, sin ruta de entrenamiento."
keywords: [Depth Anything V2, "estimación de profundidad monocular", "mapa de profundidad python", DPT, DINOv2, "profundidad relativa", "profundidad a partir de una sola imagen"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Leer el mapa de profundidad
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: denso (H, W), mayor = más cerca
        raw = depth.data                # tensor, sin unidad métrica ni escala entre imágenes
        normalized = depth.normalized() # reescalado a [0, 1] para visualización
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto
        # Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Instalación

Depth Anything V2 no necesita ningún extra opcional. Todo lo que importa está en la instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché local.

<code-tabs name="predict" />

`result.depth_map` lleva un mapa denso de profundidad inversa relativa: los
valores más altos significan más cerca de la cámara, y los valores no tienen
unidad métrica ni escala común entre imágenes. `save=True` escribe en disco una
visualización de ese mapa con un mapa de color; `Results.plot()` no cubre esta
familia, ya que está definido solo para normales de superficie y bordes. La
resolución de entrada debe ser divisible por 14, la rejilla de parches de DINOv2
sobre la que se construye la cabeza DPT; LibreYOLO lo comprueba antes de
ejecutar y lanza un error si no se cumple. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Cuatro tamaños de encoder, s/b/l/g, correspondientes a ViT-S/B/L/G. La tabla de
checkpoints de abajo solo enumera s, b y l; no hay ningún checkpoint Giant
publicado. Los cuatro comparten la misma resolución de entrada, así que elegir
un tamaño intercambia capacidad del encoder, no tamaño de imagen. La licencia
también es un factor: el checkpoint Small es Apache-2.0, mientras que Base y
Large son CC-BY-NC-4.0, consulta Licencia más abajo.

El entrenamiento y el fine-tuning no se ofrecen para esta familia.
`LibreDepthAnythingV2.train()` lanza `NotImplementedError` de forma
incondicional; en su lugar, convierte un checkpoint upstream compatible con
`weights/convert_depth_anything_v2_weights.py`.

## Validación

`val()` ejecuta el validador de profundidad compartido: alinea cada predicción
con su ground truth mediante una escala y un desplazamiento por mínimos
cuadrados calculados para cada imagen, y después reporta las métricas estándar
de profundidad relativa zero-shot, AbsRel, RMSE y los tres umbrales delta.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`, con `depth_map` en lugar de boxes.
[Exportación](/docs/export) enumera los argumentos que acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
