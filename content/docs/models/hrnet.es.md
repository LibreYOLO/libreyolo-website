---
title: HRNet
families: [hrnet]
seo_title: "HRNet: estimación de pose top-down en LibreYOLO"
description: "Usa HRNet en LibreYOLO para estimación de pose top-down con COCO-17. Instala, predice, valida y exporta los checkpoints W32 y W48, con licencia MIT."
lead: "HRNet es una red convolucional que mantiene un flujo de características de alta resolución mediante fusión multiescala repetida, en lugar de recuperar la resolución después de reducirla. LibreYOLO envuelve la variante oficial de pose top-down para inferencia y validación."
keywords: [HRNet, "estimación de pose humana", "pose top-down", "keypoints COCO-17", "estimación de pose python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sin fuente de personas: HRNet se empareja automáticamente con un
        # detector LibreYOLO9t ligero y registra esa elección una sola vez.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreHRNetw32-pose.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Fuente de personas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Sáltate la detección por completo: trata toda la imagen como una persona.
        result = model(SAMPLE_IMAGE, cropped=True)

        # O pásale a HRNet cajas de un detector que ya hayas ejecutado.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # O empareja HRNet con un detector LibreYOLO concreto en lugar del
        # LibreYOLO9t por defecto.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Usar el archivo exportado
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # El grafo exportado es solo la cabeza de heatmaps de lienzo fijo: recibe
        # un batch de recortes de persona ya recortados y ya normalizados y
        # devuelve heatmaps en bruto. La detección de personas, la geometría del
        # recorte, la decodificación de heatmaps y la supresión OKS no forman
        # parte de este grafo; ejecutarlo fuera de LibreYOLO implica
        # reimplementar tú mismo ese paso de decodificación.
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
---

## Instalación

HRNet no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

Su detector de personas por defecto, un checkpoint LibreYOLO9t ligero, se
descarga automáticamente la primera vez que HRNet se empareja con él.

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

HRNet es un estimador de pose top-down: necesita una caja de persona antes de
que la cabeza de pose pueda ejecutarse, así que cada llamada resuelve una. Si no
se toca nada, se empareja con un detector LibreYOLO9t la primera vez y registra
esa elección. `cropped=True` se salta la detección y trata toda la imagen como
una sola persona; `person_boxes` acepta cajas de un detector que ya hayas
ejecutado; `person_detector` acepta `"auto"`, `"rfdetr"`, cualquier modelo de
detección de LibreYOLO o un callable normal. `flip_test=True` ejecuta además el
modelo sobre el recorte volteado horizontalmente y promedia los dos heatmaps, el
aumento de datos en test (data augmentation) propio de HRNet; el `augment=True`
genérico no está definido aquí. Las fuentes con varias imágenes se procesan
secuencialmente: el detector de HRNet y el número variable de personas por
imagen no admiten predicción apilada. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Dos tamaños, `w32` y `w48`, que predicen ambos el conjunto estándar de 17
keypoints de COCO a partir de un recorte de persona de resolución fija; `w48` es
el más ancho de los dos backbones.

El model zoo upstream reporta la precisión de pose de cada tamaño con su propio
detector de personas, su propia configuración de flip-testing y el protocolo
oficial de evaluación de COCO. El emparejamiento por defecto de LibreYOLO usa un
detector distinto, así que una ejecución de validación aquí mide esa
combinación, no la de upstream; igualar las cifras de upstream exige las mismas
cajas de persona, las mismas puntuaciones del detector y la misma configuración
de flip que usó la evaluación original.

## Validación

`val()` calcula el OKS-AP de keypoints al estilo COCO y acepta un `data.yaml` de
YOLO-pose o un JSON de keypoints de COCO junto a un directorio de imágenes. El
backend de métricas es faster-coco-eval por defecto, y se usa `pycocotools`
automáticamente cuando faster-coco-eval no está instalado;
`faster_coco_eval=False` fuerza la vía de `pycocotools`.

<code-tabs name="val" />

La validación ejecuta internamente el propio `predict()` de HRNet, así que usa
el detector de personas con el que se construyó o se llamó al modelo. Construye
el modelo con un `person_detector=` explícito para mantener esa fuente fija
entre ejecuciones, en lugar de dejar que cada llamada vuelva a resolver el valor
por defecto.

## Exportación

<export-matrix />

El contrato de exportación de HRNet cubre solo ONNX, TorchScript, OpenVINO y
TensorRT; cualquier otro formato lanza un error antes de que empiece el trace.
Toda exportación es únicamente la cabeza de heatmaps de lienzo fijo, batch de
uno y FP32, que recibe un recorte de persona y devuelve heatmaps en bruto: la
geometría afín del recorte que va antes, y la decodificación de heatmaps, la
restauración del flip y la supresión OKS que van después, se quedan en Python,
así que un pipeline completo de imagen a keypoints sigue necesitando LibreYOLO
al otro extremo.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
