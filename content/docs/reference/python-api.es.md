---
title: API de Python
seo_title: "Referencia de la API de Python de LibreYOLO"
description: "Los nombres que LibreYOLO exporta a nivel de paquete: las cinco factories, las clases de familia, los payloads de Results, backends, validadores, trackers y utilidades de datos."
lead: "La superficie pública de Python de LibreYOLO es la lista __all__ de libreyolo/__init__.py. Todo lo que aparece en esta página se puede importar con from libreyolo import <nombre>; lo que no esté en esa lista es interno."
keywords:
  - api python libreyolo
  - importar libreyolo python
  - factory de LibreYOLO
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: "1.5.0"
verification: "Nombres y firmas leídos de libreyolo/__init__.py, libreyolo/models/__init__.py, libreyolo/models/base/model.py, libreyolo/models/base/inference.py, libreyolo/models/sam/model.py, libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py y libreyolo/ensemble/model.py en la v1.5.0."
snippets:
  usage:
    - label: Cargar cualquier cosa desde una sola factory
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Una fuente de una sola imagen devuelve un Results; una lista o un
        # directorio devuelven una lista de ellos.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Importar una clase de familia directamente
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Los cinco puntos de entrada
      language: python
      code: |
        from libreyolo import LibreYOLO, LibreEnsemble

        # Factory que inspecciona los pesos, para las familias sin prompt.
        detector = LibreYOLO("LibreYOLO9t.pt")

        # Dos o más detectores tras una sola superficie de predicción.
        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Las otras tres factories necesitan un extra instalado:
        #   pip install 'libreyolo[sam]'        -> from libreyolo import LibreSAM
        #   pip install 'libreyolo[vlm]'        -> from libreyolo import LibreVLM
        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import LibreOpenVocab
        print(type(detector).__name__, ens.fusion)
---

## Puntos de entrada

Cinco callables cargan un modelo. Se separan por contrato de llamada, no por
arquitectura.

| Factory | Qué carga | Prompt en la llamada | Extra necesario |
|---|---|---|---|
| `LibreYOLO` | Familias sin prompt, inspeccionando el checkpoint o el sufijo del archivo | | |
| `LibreSAM` | Segmentadores con prompt, por alias de tamaño | Puntos, boxes o texto de concepto | `sam` |
| `LibreVLM` | Detectores generativos de visión y lenguaje, por alias | Vocabulario de clases o un prompt libre | `vlm` |
| `LibreOpenVocab` | Detectores condicionados por texto, por alias | Vocabulario de clases | `openvocab` |
| `LibreEnsemble` | Dos o más detectores, fusionados en una sola superficie | | |

<code-tabs name="factories" />

`LibreYOLO` es la única que lee un archivo. Las otras tres reciben un alias en
forma de cadena y lo resuelven a un repositorio de Hugging Face, así que el
argumento es un nombre de modelo y no una ruta.

```python
LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
)
```

`model_path` acepta un checkpoint `.pt`, un archivo ONNX `.onnx`, un ExecuTorch
`.pte`, un MNN `.mnn`, un TensorRT `.engine`, un directorio de OpenVINO, Paddle
o ncnn, o una URL HTTP o HTTPS de un modelo en Triton. `size` y `nb_classes` se
leen del checkpoint cuando se omiten. `compute_units` solo se lee al cargar un
CoreML `.mlpackage` y toma uno de estos valores: `all`, `cpu_only`,
`cpu_and_gpu`, `cpu_and_ne`. `task` acepta cualquier nombre canónico de tarea de
`libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Clases de familia

Todas las familias que la factory puede devolver se exportan también por nombre,
así que se puede construir una clase directamente cuando el checkpoint se conoce
de antemano. Los constructores siguen a `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size` no tiene valor por defecto en una clase de familia, y esa es la
diferencia con la factory. YOLO9 y sus variantes insertan `reg_max: int = 16`
después de `size`.

Familias de detección y multitarea: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Familias de predicción densa: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Familias de clasificación y embeddings: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Otras tareas: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Los niveles hermanos también exportan sus clases de familia: `LibreSAM1`,
`LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (también escrito
`LibreModus`).

## Superficie de predicción

Llamar a un modelo ejecuta la inferencia. `predict` es un alias de `__call__`,
así que ambos son intercambiables.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

Una fuente de una sola imagen devuelve un `Results`. Una lista, una tupla o un
directorio devuelven una lista de ellos, y `stream=True` devuelve un generador.
Los demás métodos del objeto del modelo están documentados en la
[página de la API del modelo](/docs/reference/model-api).

## Payloads de Results

`Results` y sus dieciocho clases de payload se exportan a nivel de paquete:
`Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`,
`SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`,
`RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`.
Cada una se describe en [Tipos de Results](/docs/reference/results-types).

## Backends

Los artefactos exportados se cargan con `LibreYOLO()` según el sufijo del
archivo, así que las clases de backend rara vez se construyen a mano. Se
exportan para los casos en los que hay que seleccionar un backend
explícitamente: `OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`,
`TensorRTBackend`, `TritonBackend`, `NcnnBackend`, `CoreMLBackend`, además de
`create_triton_config`. `BaseExporter` es el registro de exportadores que hay
detrás de `model.export()`.

## Validadores

`model.val()` despacha al validador correcto según la tarea, así que estos se
exportan para usarlos directamente y para heredar de ellos:
`DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator`, y el `ValidationConfig` compartido.

## Seguimiento

`model.track()` selecciona un tracker por nombre. Las clases de tracker y sus
dataclasses de configuración también se exportan: `ByteTracker` con
`TrackConfig`, `BoTSortTracker` con `BoTSortConfig`, y `OCSortTracker` con
`OCSortConfig`.

## Utilidades de datos

`DATASETS_DIR` es la raíz de datasets ya resuelta, `load_data_config` lee un
YAML de dataset y `check_dataset` valida uno. Los loaders específicos de cada
tarea que se nombran en [Formatos de dataset](/docs/reference/dataset-formats)
viven en `libreyolo.data` y no a nivel de paquete.

## Galerías y destilación

`Gallery` y `FaceGallery` guardan los vectores de identidad registrados para la
tarea `embed` y producen el payload `Identities`. `Distiller` y
`get_distill_config` dirigen el entrenamiento teacher-student.

## Assets

`SAMPLE_IMAGE` es una ruta absoluta a una imagen incluida en el paquete, así que
todos los snippets de esta documentación se ejecutan sin descargar antes ninguna
imagen.

## Imports lazy y clases renombradas

La mayoría de los nombres de los niveles hermanos, los backends, los validadores
y las utilidades de datos se resuelven a través del `__getattr__` a nivel de
módulo, así que importar `libreyolo` no importa sus dependencias. El import
sigue fallando con un mensaje claro cuando falta el extra necesario.

Dos clases se renombraron y el nombre antiguo se sigue resolviendo, con un
`DeprecationWarning`: `LibreYOLORTDETR` ahora es `LibreRTDETR`, y
`LibreYOLORFDETR` ahora es `LibreRFDETR`.
