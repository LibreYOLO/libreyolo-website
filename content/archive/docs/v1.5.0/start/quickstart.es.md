---
title: Inicio rápido
seo_title: Inicio rápido de LibreYOLO
description: >-
  Ejecuta un detector sobre una imagen, haz fine-tuning con un dataset pequeño y
  expórtalo a TorchScript u ONNX, todo en CPU, en unas diez líneas de Python.
lead: >-
  El camino más corto a través de LibreYOLO: predice sobre una imagen, entrena
  con un dataset pequeño y luego exporta el resultado. Todos los comandos de
  esta página se ejecutan en CPU.
keywords:
  - libreyolo inicio rápido
  - tutorial libreyolo
  - libreyolo predict
  - entrenar libreyolo
  - exportar libreyolo onnx
  - ejemplo yolo python
last_verified: 1.5.0
meta:
  - label: Instalación
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Hardware
    value: Con la CPU basta para todo lo de esta página
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Descarga el checkpoint en el primer uso y luego lo cachea en weights/.
        model = LibreYOLO("LibreYOLO9t.pt")

        # Una sola imagen devuelve un objeto Results.
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vídeo y streams
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True genera un Results por frame en lugar de construir una
        lista.

        # Sustituye la ruta por el índice de una webcam, una URL RTSP o una
        carpeta.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco8 es un dataset de 8 imágenes incluido con la biblioteca. Se
        descarga

        # desde una URL en el primer uso, así que no hay que ejecutar ningún
        script.

        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )


        print(results["save_dir"])

        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Validar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() devuelve un dict simple, no un objeto.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() devuelve la ruta que escribió.
        path = model.export(format="torchscript")
        print(path)

        # La factoría enruta según el sufijo del archivo, así que el artefacto
        # se carga igual que un checkpoint y devuelve el mismo objeto Results.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Instalación

```bash
pip install libreyolo
```

Eso es todo lo que necesitan las secciones de predicción y entrenamiento de
abajo. Exportar a ONNX añade un extra; consulta [instalación](/docs/install)
para la lista completa.

## Predicción

<code-tabs name="predict" />

`LibreYOLO()` es una factoría. Lee el archivo, deduce a qué familia pertenecen
los pesos y devuelve el modelo de esa familia, así que cambiar a otro detector
es un cambio de una línea. Pasar `LibreYOLO9t.pt` sin directorio busca
`weights/LibreYOLO9t.pt` relativo al directorio de trabajo y lo descarga ahí
cuando falta. Consulta [checkpoints y pesos](/docs/weights) para las reglas de
descarga y cómo trabajar sin conexión.

`save=True` escribe una copia anotada bajo `runs/detect/`, en un directorio
`predict` que se incrementa en cada ejecución. El `Results` devuelto lleva
`boxes`, y `names` asocia cada índice de clase con su etiqueta. La ruta de una
sola imagen devuelve un `Results`; un directorio, una lista de imágenes o
`stream=True` devuelven una lista o un generador de ellos.

## Entrenamiento

<code-tabs name="train" />

`data` es un YAML de dataset. `coco8.yaml` viene con la biblioteca, y por eso
el snippet funciona tal cual se pega; un nombre que no viene incluido se
interpreta como una ruta. Los datasets se resuelven bajo `~/datasets`, o bajo
`LIBREYOLO_DATASETS_DIR` cuando esa variable está definida.

Una ejecución escribe en `project/name`, por defecto un directorio bajo
`runs/train`, con `weights/best.pt` y `weights/last.pt` dentro. `train()`
devuelve un diccionario que incluye `save_dir`, `best_checkpoint`,
`last_checkpoint`, las losses (funciones de pérdida) por época y las métricas
de validación por época. El checkpoint entrenado se carga mediante
`LibreYOLO()` exactamente igual que el preentrenado.

No todas las familias son entrenables. Cuando una familia solo incluye
inferencia, `train()` lanza `NotImplementedError` y lo indica.
[Conceptos básicos](/docs/concepts) explica qué significa cada nivel de
soporte.

## Exportación

<code-tabs name="export" />

TorchScript no necesita nada más allá de la instalación base. Los demás
destinos tienen cada uno su propio extra, y la cobertura es por familia y por
tarea, no uniforme: consulta [exportación y despliegue](/docs/export).

Los argumentos que acepta todo formato incluyen `imgsz` (un int, o un par de
alto y ancho), `batch` (por defecto 1), `half`, `int8` con un YAML de `data`
para la calibración, `dynamic` (por defecto True), `simplify` (por defecto
True), `opset`, `device` y `output_path`. Cuando se omite `output_path`, el
archivo se escribe bajo `weights/` con un nombre derivado del checkpoint.

## Próximos pasos

- [Conceptos básicos](/docs/concepts) para tareas, familias, tamaños y nombres de checkpoints.
- [Checkpoints y pesos](/docs/weights) para la descarga automática, el uso sin conexión y la seguridad al cargar.
- [Importar pesos existentes](/docs/migrate) si ya tienes un checkpoint de un proyecto upstream.
- [Todos los modelos](/docs/models) para la familia que encaja con tu problema.
- [Entrenamiento](/docs/train), [Predicción](/docs/predict) y [Exportación](/docs/export) para los flujos de trabajo completos.
