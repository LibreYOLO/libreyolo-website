---
title: API Python
seo_title: Référence de l'API Python LibreYOLO
description: "Noms exportés par LibreYOLO au niveau du package\_: cinq fabriques, classes de familles, charges utiles Results, backends, validateurs, trackers et assistants de données."
lead: "L'interface Python publique de LibreYOLO est la liste __all__ de libreyolo/__init__.py. Tout ce qui figure sur cette page est importable avec from libreyolo import <name>\_; tout nom absent de cette liste est interne."
keywords:
  - api python libreyolo
  - import libreyolo
  - fabrique LibreYOLO
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Noms et signatures lus dans libreyolo/__init__.py,
  libreyolo/models/__init__.py, libreyolo/models/base/model.py,
  libreyolo/models/base/inference.py, libreyolo/models/sam/model.py,
  libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py et
  libreyolo/ensemble/model.py en v1.5.0.
snippets:
  usage:
    - label: Tout charger avec une seule fabrique
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # Une source image unique renvoie un objet Results ; une liste ou un
        répertoire

        # en renvoie une liste.

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)

        print(result.names)
    - label: Importer directement une classe de famille
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Les cinq points d'entrée
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Fabrique des familles sans prompt qui inspecte les poids.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Au moins deux détecteurs derrière une interface de prédiction unique.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Les trois autres fabriques exigent l'installation d'un extra :

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Points d'entrée

Cinq callables chargent un modèle. Ils sont séparés selon leur contrat d'appel
et non leur architecture.

| Fabrique | Charge | Prompt au moment de l'appel | Extra requis |
|---|---|---|---|
| `LibreYOLO` | Familles sans prompt, en inspectant le checkpoint ou le suffixe du fichier | | |
| `LibreSAM` | Segmenteurs guidables, par alias de taille | Points, bounding boxes ou texte conceptuel | `sam` |
| `LibreVLM` | Détecteurs vision-langage génératifs, par alias | Vocabulaire de classes ou prompt libre | `vlm` |
| `LibreOpenVocab` | Détecteurs conditionnés par du texte, par alias | Vocabulaire de classes | `openvocab` |
| `LibreEnsemble` | Au moins deux détecteurs, fusionnés derrière une interface | | |

<code-tabs name="factories" />

`LibreYOLO` est la seule fabrique qui lit un fichier. Les trois autres
acceptent un alias textuel et le résolvent en dépôt Hugging Face. L'argument
est donc un nom de modèle et non un chemin.

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

`model_path` accepte un checkpoint `.pt`, un fichier ONNX `.onnx`, un fichier
ExecuTorch `.pte`, MNN `.mnn` ou TensorRT `.engine`, un répertoire OpenVINO,
Paddle ou ncnn, ou une URL de modèle Triton HTTP ou HTTPS. `size` et
`nb_classes` sont lus dans le checkpoint lorsqu'ils sont omis. `compute_units`
n'est lu que pour le chargement de packages CoreML `.mlpackage` et accepte
`all`, `cpu_only`, `cpu_and_gpu` ou `cpu_and_ne`. `task` accepte tout nom de
tâche canonique de `libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Classes de familles

Chaque famille que la fabrique peut renvoyer est également exportée par nom.
Vous pouvez ainsi construire directement une classe lorsque le checkpoint est
connu à l'avance. Les constructeurs suivent `BaseModel.__init__`\u00a0:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size` ne possède aucune valeur par défaut sur une classe de famille, ce qui
la distingue de la fabrique. YOLO9 et ses variantes insèrent
`reg_max: int = 16` après `size`.

Familles de détection et multitâches\u00a0: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Familles de prédiction dense\u00a0: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Familles de classification et d'embeddings\u00a0: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Autres tâches\u00a0: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Les niveaux frères exportent également leurs classes de familles\u00a0: `LibreSAM1`,
`LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`\u00a0;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`\u00a0; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (également orthographié
`LibreModus`).

## Interface de prédiction

L'appel d'un modèle exécute l'inférence. `predict` est un alias de `__call__`,
les deux sont donc interchangeables.

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

Une source image unique renvoie un objet `Results`. Une liste, un tuple ou un
répertoire en renvoie une liste, tandis que `stream=True` renvoie un
générateur. Les autres méthodes de l'objet modèle sont documentées sur la page
de l'[API du modèle](/docs/reference/model-api).

## Charges utiles Results

`Results` et ses dix-huit classes de charges utiles sont exportés au niveau du
package\u00a0: `Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`,
`Gaze`, `SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`,
`NormalMap`, `RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`,
`Identities`. Chacune est décrite dans les
[types de résultats](/docs/reference/results-types).

## Backends

Les artefacts exportés se chargent dans `LibreYOLO()` selon le suffixe du
fichier, les classes de backends sont donc rarement construites manuellement.
Elles sont exportées pour les cas où un backend doit être choisi explicitement\u00a0:
`OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`, `TensorRTBackend`,
`TritonBackend`, `NcnnBackend`, `CoreMLBackend`, ainsi que
`create_triton_config`. `BaseExporter` est le registre d'exporteurs utilisé
par `model.export()`.

## Validateurs

`model.val()` sélectionne le validateur adapté à la tâche. Les éléments
suivants sont donc exportés pour un usage direct et pour la création de
sous-classes\u00a0: `DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator` et le `ValidationConfig` partagé.

## Suivi

`model.track()` sélectionne un tracker par nom. Les classes de trackers et
leurs dataclasses de configuration sont également exportées\u00a0: `ByteTracker`
avec `TrackConfig`, `BoTSortTracker` avec `BoTSortConfig` et `OCSortTracker`
avec `OCSortConfig`.

## Assistants de données

`DATASETS_DIR` est la racine résolue des datasets, `load_data_config` lit un
fichier YAML de dataset et `check_dataset` le valide. Les chargeurs propres aux
tâches nommés dans les [formats de datasets](/docs/reference/dataset-formats)
se trouvent dans `libreyolo.data` plutôt qu'au niveau du package.

## Galeries et distillation

`Gallery` et `FaceGallery` contiennent les vecteurs d'identités inscrites pour
la tâche `embed` et produisent la charge utile `Identities`. `Distiller` et
`get_distill_config` pilotent l'entraînement professeur-élève.

## Assets

`SAMPLE_IMAGE` est un chemin absolu vers une image fournie avec le package.
Chaque extrait de cette documentation s'exécute donc sans devoir d'abord
télécharger une image.

## Imports différés et classes renommées

La plupart des noms des niveaux frères, les backends, les validateurs et les
assistants de données sont résolus par `__getattr__` au niveau du module.
Importer `libreyolo` n'importe donc pas leurs dépendances. L'import échoue
tout de même avec un message explicite lorsque l'extra requis manque.

Deux classes ont été renommées et leur ancienne orthographe continue de se
résoudre avec un `DeprecationWarning`\u00a0: `LibreYOLORTDETR` s'appelle désormais
`LibreRTDETR` et `LibreYOLORFDETR` s'appelle désormais `LibreRFDETR`.
