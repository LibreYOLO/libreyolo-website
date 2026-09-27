---
title: Python-API
seo_title: LibreYOLO-Python-API-Referenz
description: >-
  Die von LibreYOLO auf Paketebene exportierten Namen: fünf Factories,
  Familienklassen, Results-Payloads, Backends, Validatoren, Tracker und
  Datenhilfsfunktionen.
lead: >-
  Die öffentliche Python-Oberfläche von LibreYOLO ist die __all__-Liste in
  libreyolo/__init__.py. Alles auf dieser Seite kann als from libreyolo import
  <name> importiert werden. Alles andere ist intern.
keywords:
  - libreyolo python api
  - libreyolo import
  - LibreYOLO factory
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Namen und Signaturen aus libreyolo/__init__.py, libreyolo/models/__init__.py,
  libreyolo/models/base/model.py, libreyolo/models/base/inference.py,
  libreyolo/models/sam/model.py, libreyolo/models/vlm/__init__.py,
  libreyolo/models/openvocab/__init__.py und libreyolo/ensemble/model.py für
  v1.5.0.
snippets:
  usage:
    - label: Beliebiges Modell über eine Factory laden
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Eine einzelne Bildquelle gibt ein Results zurück, eine Liste oder ein
        # Verzeichnis dagegen eine Liste solcher Objekte.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Familienklasse direkt importieren
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Die fünf Einstiegspunkte
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Factory mit Gewichtserkennung für promptlose Familien.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Zwei oder mehr Detektoren hinter einer Vorhersageschnittstelle.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Die drei anderen Factories benötigen ein installiertes Extra:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Einstiegspunkte

Fünf Callables laden ein Modell. Sie sind nach Aufrufvertrag und nicht nach
Architektur getrennt.

| Factory | Lädt | Prompt beim Aufruf | Erforderliches Extra |
|---|---|---|---|
| `LibreYOLO` | Promptlose Familien durch Untersuchung des Checkpoints oder Dateisuffixes | | |
| `LibreSAM` | Promptbasierte Segmentierer anhand eines Größenalias | Punkte, Boxen oder Konzepttext | `sam` |
| `LibreVLM` | Generative Vision-Language-Detektoren anhand eines Alias | Klassenvokabular oder freier Prompt | `vlm` |
| `LibreOpenVocab` | Textkonditionierte Detektoren anhand eines Alias | Klassenvokabular | `openvocab` |
| `LibreEnsemble` | Zwei oder mehr zu einer Oberfläche fusionierte Detektoren | | |

<code-tabs name="factories" />

Nur `LibreYOLO` liest eine Datei. Die anderen drei nehmen einen Stringalias
entgegen und lösen ihn zu einem Hugging-Face-Repository auf. Das Argument ist
daher ein Modellname und kein Pfad.

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

`model_path` akzeptiert einen `.pt`-Checkpoint, eine ONNX-Datei `.onnx`, eine
ExecuTorch-Datei `.pte`, eine MNN-Datei `.mnn`, eine TensorRT-Datei `.engine`,
ein OpenVINO-, Paddle- oder ncnn-Verzeichnis oder eine Triton-HTTP- oder
HTTPS-Modell-URL. Wenn `size` und `nb_classes` fehlen, werden sie aus dem
Checkpoint gelesen. `compute_units` wird nur beim Laden eines CoreML-
`.mlpackage` gelesen und akzeptiert `all`, `cpu_only`, `cpu_and_gpu` sowie
`cpu_and_ne`. `task` nimmt jeden kanonischen Aufgabennamen aus
`libreyolo.tasks.TASKS` entgegen.

<code-tabs name="usage" />

## Familienklassen

Jede von der Factory zurückgebbare Familie wird auch unter ihrem Namen
exportiert. Wenn der Checkpoint im Voraus bekannt ist, kann die Klasse daher
direkt erstellt werden. Die Konstruktoren folgen `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

Bei einer Familienklasse besitzt `size` keinen Standardwert. Dies unterscheidet
sie von der Factory. YOLO9 und seine Varianten fügen nach `size` den Parameter
`reg_max: int = 16` ein.

Erkennungs- und Multi-Task-Familien: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`, `LibreEC`,
`LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Familien für dichte Vorhersagen: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Klassifikations- und Embedding-Familien: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Andere Aufgaben: `LibreHRNet` (Pose), `LibreL2CS` (Blick), `LibrePPOCR` (OCR),
`LibreFaceEmbedder` (Embedding).

Auch die benachbarten Stufen exportieren ihre Familienklassen: `LibreSAM1`,
`LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (auch `LibreModus`).

## Vorhersageschnittstelle

Das Aufrufen eines Modells führt die Inferenz aus. `predict` ist ein Alias für
`__call__`. Beide sind daher austauschbar.

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

Eine einzelne Bildquelle gibt ein `Results` zurück. Eine Liste, ein Tupel oder
ein Verzeichnis gibt eine Liste zurück, und `stream=True` einen Generator. Die
anderen Methoden des Modellobjekts werden auf der
[Modell-API-Seite](/docs/reference/model-api) beschrieben.

## Results-Payloads

`Results` und seine 18 Payload-Klassen werden auf Paketebene exportiert:
`Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`,
`SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`,
`RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`.
Alle werden unter [Results-Typen](/docs/reference/results-types) beschrieben.

## Backends

Exportierte Artefakte werden anhand ihres Dateisuffixes über `LibreYOLO()`
geladen. Backend-Klassen müssen daher selten von Hand erstellt werden. Sie
werden für Fälle exportiert, in denen ein Backend explizit gewählt werden muss:
`OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`, `TensorRTBackend`,
`TritonBackend`, `NcnnBackend`, `CoreMLBackend` sowie `create_triton_config`.
`BaseExporter` ist das Exporter-Register hinter `model.export()`.

## Validatoren

`model.val()` leitet nach Aufgabe an den passenden Validator weiter. Die
folgenden Klassen werden für die direkte Verwendung und Unterklassenbildung
exportiert: `DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator` und die gemeinsame `ValidationConfig`.

## Tracking

`model.track()` wählt einen Tracker anhand seines Namens. Trackerklassen und
ihre Konfigurations-Dataclasses werden ebenfalls exportiert: `ByteTracker` mit
`TrackConfig`, `BoTSortTracker` mit `BoTSortConfig` und `OCSortTracker` mit
`OCSortConfig`.

## Datenhilfsfunktionen

`DATASETS_DIR` ist der aufgelöste Datensatzstamm. `load_data_config` liest eine
Datensatz-YAML, und `check_dataset` validiert sie. Die unter
[Datensatzformate](/docs/reference/dataset-formats) benannten
aufgabenspezifischen Loader liegen in `libreyolo.data` und nicht auf
Paketebene.

## Galerien und Distillation

`Gallery` und `FaceGallery` halten registrierte Identitätsvektoren für die
Aufgabe `embed` und erzeugen die Payload `Identities`. `Distiller` und
`get_distill_config` steuern das Teacher-Student-Training.

## Assets

`SAMPLE_IMAGE` ist ein absoluter Pfad zu einem mit dem Paket ausgelieferten
Bild. Jedes Snippet dieser Dokumentation funktioniert daher, ohne zuerst ein
Bild herunterzuladen.

## Verzögerte Importe und umbenannte Klassen

Die meisten Namen benachbarter Stufen, Backends, Validatoren und
Datenhilfsfunktionen werden über `__getattr__` auf Modulebene aufgelöst. Beim
Import von `libreyolo` werden ihre Abhängigkeiten daher nicht importiert. Wenn
das erforderliche Extra fehlt, scheitert der spätere Import weiterhin mit
einer verständlichen Meldung.

Zwei Klassennamen wurden umbenannt. Die alte Schreibweise wird weiterhin mit
einem `DeprecationWarning` aufgelöst: `LibreYOLORTDETR` heißt jetzt
`LibreRTDETR`, und `LibreYOLORFDETR` heißt jetzt `LibreRFDETR`.

