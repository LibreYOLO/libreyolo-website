---
title: Depth Anything 3
families: [depth_anything3]
seo_title: "Depth Anything 3: predice profundidad monocular en LibreYOLO"
description: "Usa Depth Anything 3 en LibreYOLO para estimación de profundidad monocular. Instala, predice, valida y exporta el checkpoint DA3MONO-LARGE, Apache-2.0."
lead: "Depth Anything 3 es un transformer DINOv2 sin más, entrenado para predecir profundidad y geometría de cámara a partir de una o varias vistas sin ninguna especialización arquitectónica. LibreYOLO porta su checkpoint DA3MONO-LARGE para la tarea de profundidad: predicción y validación zero-shot, sin ruta de entrenamiento."
keywords: [Depth Anything 3, DA3, "estimación de profundidad monocular", DINOv2, "profundidad relativa", "mapa de profundidad python", "estimar profundidad de una imagen"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDepthAnything3l-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Leer el mapa de profundidad
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map    # DepthMap: densa (H, W), más alto = más cerca
        raw = depth.data                # tensor, sin unidad métrica ni escala entre imágenes
        normalized = depth.normalized() # reescalado a [0, 1] para visualización
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx
        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
---

## Instalación

Depth Anything 3 no necesita ningún extra opcional. Todo lo que importa está en
la instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

`result.depth_map` lleva un mapa denso de profundidad inversa relativa: los
valores más altos significan más cerca de la cámara, y los valores no tienen
unidad métrica ni escala común entre imágenes. El checkpoint original emite
profundidad relativa positiva; el wrapper de red de LibreYOLO la invierte y
reproduce el tratamiento oficial del cielo para que la salida siga el contrato de
profundidad compartido de LibreYOLO. `save=True` escribe en disco una
visualización de ese mapa con mapa de color; `Results.plot()` no cubre esta
familia, ya que está definido solo para normales de superficie y bordes. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Un solo tamaño, `l`, con una resolución de entrada fija. El proyecto original de
DA3 publica además checkpoints any-view Small y Base, un checkpoint de
profundidad métrica y checkpoints Nested y Giant; LibreYOLO no expone ninguno de
ellos. La profundidad métrica necesita un contrato público distinto del de la
tarea de profundidad inversa relativa de LibreYOLO, y los checkpoints any-view y
Nested necesitan una API de cámara multiimagen que LibreYOLO no ofrece. Los
checkpoints any-view Large y Giant son además CC-BY-NC-4.0 y ninguna ruta de
descarga de LibreYOLO los referencia.

Esta familia no ofrece entrenamiento. `LibreDepthAnything3.train()` lanza
`NotImplementedError` incondicionalmente; entrena con el proyecto original y
convierte un checkpoint DA3MONO-LARGE compatible con
`weights/convert_depth_anything3_weights.py`.

## Validación

`val()` ejecuta el validador de profundidad compartido: alinea cada predicción
con su ground truth mediante una escala y un desplazamiento por mínimos cuadrados
calculados por imagen, y después informa de las métricas estándar de profundidad
relativa zero-shot: AbsRel, RMSE y los tres umbrales delta.

<code-tabs name="val" />

## Exportación

<export-matrix />

En esta familia la exportación está restringida a cinco formatos: ONNX,
TorchScript, ExecuTorch, TensorRT y OpenVINO. Pedir cualquier otro formato lanza
`NotImplementedError` en lugar de intentar una conversión sin validar. Un
artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión de
archivo, así que un archivo `.onnx` o `.engine` se comporta como un checkpoint y
devuelve el mismo `Results`, con `depth_map` en lugar de boxes.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
