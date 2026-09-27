---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: estimación de profundidad monocular en LibreYOLO'
description: >-
  Usa MiDaS en LibreYOLO para estimación de profundidad monocular. Instala,
  predice, valida y exporta dos variantes con licencia MIT, descargadas desde
  isl-org.
lead: >-
  MiDaS es estimación de profundidad relativa monocular entrenada con una loss
  invariante a escala y desplazamiento sobre datasets mezclados, la línea de
  trabajo que estableció el protocolo de transferencia de profundidad zero-shot
  que reutilizan las familias posteriores. LibreYOLO lo soporta para la tarea de
  profundidad: predicción y validación zero-shot, sin ruta de entrenamiento.
keywords:
  - MiDaS
  - estimación de profundidad monocular
  - DPT
  - mapa de profundidad python
  - profundidad relativa
  - profundidad zero-shot
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Aún no está en disco: LibreYOLO lo descarga de la release oficial de

        # isl-org/MiDaS en GitHub y lo comprueba contra un SHA-256 fijado antes
        de usarlo.

        model = LibreYOLO("LibreMiDaSl-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)


        depth = result.depth_map

        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Variante Small
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Encoder EfficientNet-Lite3, más pequeño y rápido que el tamaño l,
        DPT-Large.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("LibreMiDaSl-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## Instalación

MiDaS no necesita ningún extra opcional. Todo lo que importa está en la instalación base.

```bash
pip install libreyolo
```

## Predicción

MiDaS es la única familia de profundidad que LibreYOLO no republica en su propia
organización de Hugging Face. Pedir un checkpoint por su nombre de archivo de
LibreYOLO descarga el asset oficial correspondiente directamente de las releases
de GitHub de `isl-org/MiDaS`, lo comprueba contra un SHA-256 fijado y lo envuelve
con los metadatos de checkpoint de LibreYOLO antes del primer uso; las
ejecuciones posteriores reutilizan el archivo local en caché. Consulta Licencia
para saber por qué.

<code-tabs name="predict" />

`result.depth_map` lleva un mapa denso de profundidad inversa relativa: los
valores más altos significan más cerca de la cámara, y los valores no tienen
unidad métrica ni escala común entre imágenes. `save=True` escribe en disco una
visualización de ese mapa con un mapa de color; `Results.plot()` no cubre esta
familia, ya que está definido solo para normales de superficie y bordes.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Dos variantes con encoders diferentes, no solo escalas distintas del mismo
encoder. `s` es MiDaS v2.1 Small, un encoder EfficientNet-Lite3. `l` es
DPT-Large, un encoder ViT-L/16 con el decoder DPT que MiDaS introdujo para
predicción densa. También preprocesan de forma distinta: `s` usa un redimensionado
con cota superior de aspecto y normalización con la media/desviación típica de
ImageNet, `l` usa un redimensionado de aspecto mínimo con media y desviación
típica de 0.5. Elige `s` para una CNN más ligera, `l` para la precisión del
decoder transformer.

El entrenamiento no se ofrece para esta familia. `LibreMiDaS.train()` lanza
`NotImplementedError` de forma incondicional.

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

<code-tabs name="export" />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
