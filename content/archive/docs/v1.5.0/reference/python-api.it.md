---
title: API Python
seo_title: Riferimento dell'API Python di LibreYOLO
description: >-
  I nomi che LibreYOLO esporta a livello di pacchetto: le cinque factory, le
  classi di famiglia, i payload di Results, i backend, i validatori, i tracker e
  le utility per i dati.
lead: >-
  La superficie pubblica Python di LibreYOLO è la lista __all__ in
  libreyolo/__init__.py. Tutto quello che sta in questa pagina è importabile con
  from libreyolo import <nome>; qualsiasi cosa non sia in quella lista è
  interna.
keywords:
  - api python libreyolo
  - importare libreyolo python
  - factory LibreYOLO
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Nomi e firme letti da libreyolo/__init__.py, libreyolo/models/__init__.py,
  libreyolo/models/base/model.py, libreyolo/models/base/inference.py,
  libreyolo/models/sam/model.py, libreyolo/models/vlm/__init__.py,
  libreyolo/models/openvocab/__init__.py e libreyolo/ensemble/model.py alla
  v1.5.0.
snippets:
  usage:
    - label: Caricare qualsiasi cosa da un'unica factory
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Una sorgente con una sola immagine restituisce un Results; una lista
        # o una directory ne restituiscono una lista.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Importare direttamente una classe di famiglia
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: I cinque punti di ingresso
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Factory che annusa i pesi, sulle famiglie senza prompt.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Due o più rilevatori dietro un'unica superficie di predizione.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Le altre tre factory richiedono un extra installato:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Punti di ingresso

Cinque callable caricano un modello. Sono separati dal contratto di chiamata,
non dall'architettura.

| Factory | Cosa carica | Prompt al momento della chiamata | Extra necessario |
|---|---|---|---|
| `LibreYOLO` | Famiglie senza prompt, annusando il checkpoint o il suffisso del file | | |
| `LibreSAM` | Segmentatori con prompt, per alias di dimensione | Punti, box o testo di concetto | `sam` |
| `LibreVLM` | Rilevatori generativi vision-language, per alias | Vocabolario di classi o un prompt libero | `vlm` |
| `LibreOpenVocab` | Rilevatori condizionati dal testo, per alias | Vocabolario di classi | `openvocab` |
| `LibreEnsemble` | Due o più rilevatori, fusi in un'unica superficie | | |

<code-tabs name="factories" />

`LibreYOLO` è l'unica che legge un file. Le altre tre prendono un alias sotto
forma di stringa e lo risolvono in un repository di Hugging Face, quindi
l'argomento è il nome di un modello e non un percorso.

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

`model_path` accetta un checkpoint `.pt`, un file ONNX `.onnx`, un ExecuTorch
`.pte`, un MNN `.mnn`, un TensorRT `.engine`, una directory OpenVINO, Paddle o
ncnn, oppure l'URL HTTP o HTTPS di un modello su Triton. `size` e `nb_classes`
vengono letti dal checkpoint quando sono omessi. `compute_units` viene letto
solo per i caricamenti di un CoreML `.mlpackage` ed è uno tra `all`,
`cpu_only`, `cpu_and_gpu`, `cpu_and_ne`. `task` accetta qualsiasi nome
canonico di task da `libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Classi di famiglia

Ogni famiglia che la factory può restituire è esportata anche per nome, così si
può costruire una classe direttamente quando il checkpoint è noto in anticipo.
I costruttori seguono `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size` non ha un valore predefinito in una classe di famiglia, ed è questa la
differenza rispetto alla factory. YOLO9 e le sue varianti inseriscono
`reg_max: int = 16` dopo `size`.

Famiglie di rilevamento e multi-task: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Famiglie di predizione densa: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Famiglie di classificazione ed embedding: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Altri task: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Anche i tier gemelli esportano le loro classi di famiglia: `LibreSAM1`,
`LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (scritto anche
`LibreModus`).

## Superficie di predizione

Chiamare un modello esegue l'inferenza. `predict` è un alias di `__call__`,
quindi i due sono intercambiabili.

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

Una sorgente con una sola immagine restituisce un `Results`. Una lista, una
tupla o una directory ne restituiscono una lista, e `stream=True` restituisce
un generatore. Gli altri metodi dell'oggetto modello sono documentati nella
[pagina dell'API del modello](/docs/reference/model-api).

## Payload di Results

`Results` e le sue diciotto classi di payload sono esportate a livello di
pacchetto: `Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`,
`Gaze`, `SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`,
`NormalMap`, `RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`,
`Identities`. Ognuna è descritta in
[Tipi di Results](/docs/reference/results-types).

## Backend

Gli artefatti esportati si caricano attraverso `LibreYOLO()` in base al
suffisso del file, quindi le classi di backend si costruiscono raramente a
mano. Sono esportate per i casi in cui un backend va selezionato
esplicitamente: `OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`,
`TensorRTBackend`, `TritonBackend`, `NcnnBackend`, `CoreMLBackend`, più
`create_triton_config`. `BaseExporter` è il registro degli exporter dietro
`model.export()`.

## Validatori

`model.val()` smista verso il validatore giusto in base al task, quindi questi
sono esportati per l'uso diretto e per essere estesi tramite sottoclassi:
`DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator`, e il `ValidationConfig` condiviso.

## Tracking

`model.track()` seleziona un tracker per nome. Anche le classi dei tracker e le
loro dataclass di configurazione sono esportate: `ByteTracker` con
`TrackConfig`, `BoTSortTracker` con `BoTSortConfig` e `OCSortTracker` con
`OCSortConfig`.

## Utility per i dati

`DATASETS_DIR` è la radice risolta dei dataset, `load_data_config` legge un
YAML di dataset e `check_dataset` ne convalida uno. I loader specifici per task
elencati in [Formati dei dataset](/docs/reference/dataset-formats) stanno in
`libreyolo.data` e non a livello di pacchetto.

## Gallery e distillazione

`Gallery` e `FaceGallery` conservano i vettori di identità registrati per il
task `embed` e producono il payload `Identities`. `Distiller` e
`get_distill_config` guidano l'addestramento teacher-student.

## Asset

`SAMPLE_IMAGE` è un percorso assoluto a un'immagine inclusa nel pacchetto, così
ogni snippet di questa documentazione gira senza dover prima scaricare una
foto.

## Import lazy e classi rinominate

La maggior parte dei nomi dei tier gemelli, i backend, i validatori e le
utility per i dati si risolvono attraverso il `__getattr__` a livello di
modulo, quindi importare `libreyolo` non importa le loro dipendenze. L'import
fallisce comunque con un messaggio chiaro quando manca l'extra richiesto.

Due nomi di classe sono stati rinominati e la vecchia grafia si risolve ancora,
con un `DeprecationWarning`: `LibreYOLORTDETR` ora è `LibreRTDETR`, e
`LibreYOLORFDETR` ora è `LibreRFDETR`.
