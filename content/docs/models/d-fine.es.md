---
title: D-FINE
families: [dfine]
seo_title: "D-FINE: haz fine-tuning, valida y exporta bajo MIT"
description: "Usa D-FINE en LibreYOLO para detección de objetos y segmentación de instancias. Instala, predice, haz fine-tuning, valida y exporta, todo con código bajo licencia MIT."
lead: "Un transformer de detección que reformula la regresión de cajas como una distribución de probabilidad sobre cada borde de la caja, refinada a lo largo de las capas del decoder. LibreYOLO lo soporta para detección y segmentación de instancias."
keywords: [D-FINE, "transformer de detección", "detección de objetos en tiempo real", "segmentación de instancias", "fine-tuning D-FINE", DETR]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -seg del nombre de archivo selecciona la cabeza de máscaras,
        # así que aquí no hace falta el argumento task.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Segmentación de instancias
      language: bash
      code: |
        # Continúa desde pesos de segmentación publicados, cabeza de máscaras incluida.
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentación desde pesos de detección
      language: bash
      code: |
        # Los pesos de detección no llevan cabeza de máscaras, así que esto es una
        # transferencia explícita: la cabeza empieza sin entrenar y solo sirve una
        # vez entrenada. Pedir task=segment aquí es lo que autoriza la transferencia.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # cajas
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640
        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640 half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según el sufijo del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo Results.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

D-FINE no necesita ningún extra opcional. Todos sus imports están cubiertos por
la instalación base.

```bash
pip install libreyolo
```

El fine-tuning con adaptadores mediante `lora=True` es la excepción, y necesita
el extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. Un nombre de archivo con
`-seg` resuelve por sí solo a la tarea de segmentación, y entonces
`result.masks` lleva las máscaras de instancia junto a las cajas. `conf` y
`max_det` filtran la selección de queries; `iou` se acepta por paridad de API
pero no tiene efecto, porque el decoder es un predictor de conjuntos sin paso de
NMS. Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Cinco tamaños. Todos funcionan a la misma resolución de entrada, así que la
tabla los separa por número de parámetros y precisión.

<benchmark-table task="detect" />

<va-embed />

La segmentación reutiliza el backbone, el encoder y el decoder de detección y
añade una cabeza de máscaras, así que un checkpoint `-seg` acepta los mismos
argumentos que su equivalente de detección. La familia RT-DETRv4 de LibreYOLO
está escrita como una subclase del wrapper de D-FINE: hereda esta línea de
decoder y luego fija su lista de tareas de vuelta a detección, porque no lleva
cabeza de máscaras.

## Entrenamiento

El entrenamiento parte de un checkpoint publicado, para ambas tareas.

<code-tabs name="train" />

Si no se toca nada, el trainer ejecuta 132 épocas con `lr0=2e-4` y `amp=False`,
un batch de 16 y early stopping tras 50 épocas sin mejora. Los pesos de
detección son un punto de partida válido para entrenar segmentación, pero solo
como transferencia explícita, ya que la cabeza de máscaras empieza sin entrenar
y de otro modo devolvería máscaras sin sentido. Pasar `task=segment` en el CLI
es lo que la autoriza. La vía de Python es más estrecha: hay que construir
`LibreDFINE` directamente con `allow_detect_to_segment_transfer=True`, porque la
factoría `LibreYOLO()` no acepta ese argumento, y la construcción directa no
descarga nada, así que el archivo de pesos ya tiene que estar en disco.

`lora=True` se aplica a detección. El entrenamiento de segmentación lo rechaza y
remite a `freeze='backbone'` en su lugar, porque la cabeza de máscaras no se ha
probado con adaptadores. En Apple silicon el trainer mueve toda la ejecución a
CPU: el backward pass del matmul por bins del Integral choca con un fallo de
compilación de Metal. La inferencia en MPS no se ve afectada.

Consulta [entrenamiento](/docs/train) para datasets, aumento de datos
(data augmentation), multi-GPU y loggers.

## Validación

`val()` devuelve un diccionario indexado por nombre de métrica, e imprime los
resultados por clase si `verbose` se deja activado.

<code-tabs name="val" />

Contra un checkpoint `-seg`, la clave `metrics/mAP50-95` a secas contiene la
puntuación de las máscaras, y la misma ejecución reporta además las cajas bajo
`(B)` y las máscaras bajo `(M)`, así que ambas están disponibles en una sola
pasada.

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su sufijo de
archivo, así que un archivo `.onnx` o `.engine` se comporta como un checkpoint y
devuelve el mismo `Results`. Las rutas de OpenVINO, Paddle, MNN y Core AI
exportan con un lienzo fijo en lugar de con formas dinámicas.
[Exportación](/docs/export) lista los argumentos que acepta cada formato y los
extras que añaden unos pocos de ellos.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

Los pesos de segmentación tienen un segundo upstream: su decoder de máscaras, el
emparejamiento de máscaras y la función de pérdida (loss) de máscaras vienen de
ArgoHA/D-FINE-seg, también Apache-2.0, cuyo maintainer aprobó el reuso con
atribución.

</provenance-box>

## Cita

<citation-block />
