---
title: Python API
seo_title: "LibreYOLO Python API reference"
description: "The names LibreYOLO exports at package level: the five factories, the family classes, the Results payloads, backends, validators, trackers and data helpers."
lead: "The public Python surface of LibreYOLO is the __all__ list in libreyolo/__init__.py. Everything on this page is importable as from libreyolo import <name>; anything not on that list is internal."
keywords:
  - libreyolo python api
  - libreyolo import
  - LibreYOLO factory
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: "1.5.0"
verification: "Names and signatures read from libreyolo/__init__.py, libreyolo/models/__init__.py, libreyolo/models/base/model.py, libreyolo/models/base/inference.py, libreyolo/models/sam/model.py, libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py and libreyolo/ensemble/model.py at v1.5.0."
snippets:
  usage:
    - label: Load anything through one factory
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # A single image source returns one Results; a list or directory
        # returns a list of them.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Import a family class directly
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: The five entry points
      language: python
      code: |
        from libreyolo import LibreYOLO, LibreEnsemble

        # Weight-sniffing factory over the promptless families.
        detector = LibreYOLO("LibreYOLO9t.pt")

        # Two or more detectors behind one prediction surface.
        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # The other three factories need an extra installed:
        #   pip install 'libreyolo[sam]'        -> from libreyolo import LibreSAM
        #   pip install 'libreyolo[vlm]'        -> from libreyolo import LibreVLM
        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import LibreOpenVocab
        print(type(detector).__name__, ens.fusion)
---

## Entry points

Five callables load a model. They are separated by call contract, not by
architecture.

| Factory | Loads | Prompt at call time | Extra required |
|---|---|---|---|
| `LibreYOLO` | Promptless families, by sniffing the checkpoint or file suffix | | |
| `LibreSAM` | Promptable segmenters, by size alias | Points, boxes, or concept text | `sam` |
| `LibreVLM` | Generative vision-language detectors, by alias | Class vocabulary or a free-form prompt | `vlm` |
| `LibreOpenVocab` | Text-conditioned detectors, by alias | Class vocabulary | `openvocab` |
| `LibreEnsemble` | Two or more detectors, fused into one surface | | |

<code-tabs name="factories" />

`LibreYOLO` is the only one that reads a file. The other three take a string
alias and resolve it to a Hugging Face repository, so the argument is a model
name and not a path.

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

`model_path` accepts a `.pt` checkpoint, an ONNX `.onnx` file, an ExecuTorch
`.pte`, an MNN `.mnn`, a TensorRT `.engine`, an OpenVINO, Paddle or ncnn
directory, or a Triton HTTP or HTTPS model URL. `size` and `nb_classes` are
read from the checkpoint when omitted. `compute_units` is read only for
CoreML `.mlpackage` loads and is one of `all`, `cpu_only`, `cpu_and_gpu`,
`cpu_and_ne`. `task` takes any canonical task name from `libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Family classes

Every family the factory can return is also exported by name, so a class can be
constructed directly when the checkpoint is known in advance. The constructors
follow `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size` has no default on a family class, which is the difference from the
factory. YOLO9 and its variants insert `reg_max: int = 16` after `size`.

Detection and multi-task families: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Dense-prediction families: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Classification and embedding families: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Other tasks: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

The sibling tiers export their family classes too: `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (also spelled
`LibreModus`).

## Prediction surface

Calling a model runs inference. `predict` is an alias for `__call__`, so the
two are interchangeable.

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

A single image source returns one `Results`. A list, a tuple or a directory
returns a list of them, and `stream=True` returns a generator. The other
methods on the model object are documented on the
[model API page](/docs/reference/model-api).

## Results payloads

`Results` and its eighteen payload classes are exported at package level:
`Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`,
`SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`,
`RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`.
Each one is described in [Results types](/docs/reference/results-types).

## Backends

Exported artifacts load through `LibreYOLO()` by file suffix, so the backend
classes are rarely constructed by hand. They are exported for the cases where
a backend has to be selected explicitly: `OnnxBackend`, `OpenVINOBackend`,
`PaddleBackend`, `TensorRTBackend`, `TritonBackend`, `NcnnBackend`,
`CoreMLBackend`, plus `create_triton_config`. `BaseExporter` is the exporter
registry behind `model.export()`.

## Validators

`model.val()` dispatches to the right validator by task, so these are exported
for direct use and for subclassing: `DetectionValidator`,
`SegmentationValidator`, `PoseValidator`, `SemanticValidator`,
`PanopticValidator`, `DepthValidator`, `NormalValidator`, `EdgeValidator`, and
the shared `ValidationConfig`.

## Tracking

`model.track()` selects a tracker by name. The tracker classes and their
configuration dataclasses are also exported: `ByteTracker` with `TrackConfig`,
`BoTSortTracker` with `BoTSortConfig`, and `OCSortTracker` with
`OCSortConfig`.

## Data helpers

`DATASETS_DIR` is the resolved dataset root, `load_data_config` reads a
dataset YAML, and `check_dataset` validates one. The task-specific loaders
named in [Dataset formats](/docs/reference/dataset-formats) live in
`libreyolo.data` rather than at package level.

## Galleries and distillation

`Gallery` and `FaceGallery` hold enrolled identity vectors for the `embed`
task and produce the `Identities` payload. `Distiller` and
`get_distill_config` drive teacher-student training.

## Assets

`SAMPLE_IMAGE` is an absolute path to an image bundled with the package, so
every snippet in these docs runs without downloading a picture first.

## Lazy imports and renamed classes

Most sibling-tier names, the backends, the validators and the data helpers
resolve through the module-level `__getattr__`, so importing `libreyolo` does
not import their dependencies. The import still fails with a clear message
when the required extra is missing.

Two class names were renamed and the old spelling still resolves, with a
`DeprecationWarning`: `LibreYOLORTDETR` is now `LibreRTDETR`, and
`LibreYOLORFDETR` is now `LibreRFDETR`.
