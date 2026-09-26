---
title: API du modèle
seo_title: Méthodes et signatures des objets modèles LibreYOLO
description: "Toutes les méthodes d'un modèle LibreYOLO chargé\_: predict, embed, track, val, train, export, save, quantize, info et réglages des graphes CUDA, avec les véritables valeurs par défaut."
lead: >-
  Un modèle LibreYOLO chargé est une instance de BaseModel. Cette page énumère
  les méthodes de cette instance, avec les signatures et valeurs par défaut lues
  dans libreyolo/models/base/model.py.
keywords:
  - méthodes modèle libreyolo
  - arguments predict libreyolo
  - arguments val libreyolo
  - arguments export libreyolo
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: "Signatures et valeurs par défaut lues dans libreyolo/models/base/model.py et libreyolo/models/base/inference.py en v1.5.0. Les classes de familles peuvent les limiter ou les étendre\_; train() est défini par famille et seul son wrapper cfg= partagé est documenté ici."
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True renvoie un générateur, avec un objet Results par image ou
        frame.

        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Construction

La fabrique renvoie une instance de classe de famille. Construire directement
cette classe accepte les mêmes arguments, à l'exception de `size`, qui est
requis\u00a0:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` choisit CUDA s'il est disponible, puis MPS, puis CPU. Un entier
ou une chaîne de chiffres est interprété comme un indice CUDA. `device=0` et
`device="0"` signifient donc tous deux `cuda:0`. `task` est validé selon la
liste `SUPPORTED_TASKS` de la famille. `model_path=None` construit
l'architecture et la laisse en mode entraînement, tandis qu'un `dict` charge
directement ce state dict.

## predict et \_\_call\_\_

`predict` est un alias de `__call__`.

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

| Argument | Valeur par défaut | Signification |
|---|---|---|
| `source` | `None` | Image, liste ou tuple d'images en mémoire, répertoire, fichier vidéo ou source écran comme `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Seuil de confiance |
| `iou` | `0.45` | Seuil IoU de la NMS |
| `imgsz` | `None` | Remplacement de la taille d'entrée\u00a0; `None` utilise la taille native du modèle |
| `device` | `None` | Remplacement de l'appareil pour cet appel |
| `classes` | `None` | Conserver uniquement ces identifiants de classe |
| `max_det` | `300` | Nombre maximal de détections par image |
| `augment` | `False` | Augmentation à l'inférence |
| `save` | `False` | Écrire une image ou vidéo annotée |
| `batch` | `1` | Nombre d'images par passe forward pour les sources répertoire et liste |
| `stream` | `False` | Renvoyer un générateur plutôt qu'une liste matérialisée |
| `stream_buffer` | `False` | Conserver chaque image capturée en direct plutôt que la seule plus récente |
| `vid_stride` | `1` | Traiter une image sur N d'une vidéo ou d'un écran |
| `show` | `False` | Afficher les images annotées dans une fenêtre |
| `output_path` | `None` | Chemin de sortie lorsque `save=True` |
| `color_format` | `"auto"` | Indication de format des couleurs pour les tableaux en mémoire |
| `tiling` | `False` | Inférence par tuiles pour les grandes images |
| `overlap_ratio` | `0.2` | Ratio de chevauchement des tuiles |
| `output_file_format` | `None` | `"jpg"`, `"png"` ou `"webp"` |
| `cuda_graph` | `False` | `True` capture à la première utilisation de chaque forme d'entrée, `"auto"` attend qu'une forme se répète |

Une source image unique renvoie un objet `Results`. Une liste, un tuple ou un
répertoire en renvoie une liste, tandis que `stream=True` renvoie un générateur
dans tous les cas.

Les sources en direct sont illimitées et exigent `stream=True`. `tiling` et
`augment` ne peuvent pas être combinés. L'augmentation à l'inférence lève une
erreur pour les tâches `embed`, `point` et `edge`.

<code-tabs name="usage" />

Avec `batch > 1`, les familles dont `SUPPORTS_BATCHED_PREDICT` vaut true
exécutent une passe forward empilée par groupe. `batch=1` conserve une passe
forward par image.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Wrapper pratique de `predict` qui empile chaque ligne d'embedding dans un
tenseur unique `(N_total, D)`. Le modèle doit avoir été construit avec
`task="embed"`, sinon la méthode lève `NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Produit un objet `Results` par image avec `track_id` défini. `tracker` vaut
`"bytetrack"`, `"botsort"`, `"ocsort"` ou `"deepocsort"`, et est ignoré si
`tracker_config` est fourni, car le type de configuration sélectionne le
tracker. `track_conf` correspond à `track_high_thresh` pour ByteTrack et
BoT-SORT, et à `det_thresh` pour OC-SORT et Deep OC-SORT. `output_path` utilise
par défaut `runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Renvoie un dictionnaire de métriques dont les clés dépendent de la tâche. La
détection renvoie `metrics/precision`, `metrics/recall`, `metrics/mAP50` et
`metrics/mAP50-95`. `imgsz` accepte un entier carré ou un tuple
`(height, width)` et utilise par défaut la taille d'entrée native du modèle.
`plots` est un alias de `save_plots`. `allow_download_scripts` autorise le
Python intégré que le champ `download` d'un YAML de dataset peut contenir.

`faster_coco_eval` est accepté dans `**kwargs` et vaut `True` par défaut, avec
un repli vers pycocotools lorsque le package n'est pas installé. Le backend
réellement exécuté est indiqué dans `model.last_eval_backend`.

La validation augmentée lève une erreur pour les tâches `obb` et `pose`.

## train

`train` est défini par famille, ses arguments diffèrent donc. Deux comportements
sont partagés, car la classe de base encapsule la méthode `train` de chaque
famille\u00a0:

- `cfg=` accepte le chemin d'un YAML dont les clés sont fusionnées dans l'appel. Les arguments nommés explicites l'emportent sur le fichier.
- `pretrained=False` sur une famille du groupe de couverture `g0` ou `g1` réinitialise le modèle à partir de zéro avant l'entraînement et ne peut pas être associé à `resume=True`.

La prise en compte réelle des réglages d'augmentation dépend de la famille.
Consultez la [matrice d'augmentation](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Renvoie le chemin de l'artefact écrit. `format` est résolu dans le registre des
exporteurs, où `engine` est un alias de `tensorrt` et `litert` un alias de
`tflite`. Arguments partagés par tous les exporteurs\u00a0:

| Argument | Valeur par défaut | Signification |
|---|---|---|
| `output_path` | `None` | Chemin du fichier de sortie\u00a0; généré sous `weights/` s'il est omis |
| `imgsz` | `None` | Tuple `(height, width)` ou entier unique\u00a0; taille native par défaut |
| `opset` | `None` | Version de l'opset ONNX |
| `simplify` | `True` | Exécuter la simplification du graphe ONNX |
| `dynamic` | `True` | Activer les axes dynamiques |
| `half` | `False` | Précision FP16 |
| `int8` | `False` | Précision INT8 |
| `batch` | `1` | Taille de batch intégrée à l'artefact |
| `device` | `None` | Appareil utilisé pour le traçage |
| `data` | `None` | data.yaml pour la calibration INT8 |
| `fraction` | `1.0` | Fraction du dataset de calibration à utiliser |
| `allow_download_scripts` | `False` | Autoriser le Python intégré aux téléchargements du YAML de dataset |
| `verbose` | `False` | Journalisation détaillée de l'exporteur |

Les combinaisons bloquées lèvent `NotImplementedError` lors des vérifications
préalables, avant le traçage. La couverture et ses règles figurent dans la
[matrice d'export](/docs/reference/export-matrix). En présence d'adaptateurs
LoRA actifs, ceux-ci sont fusionnés dans des poids denses, uniquement après
tous les contrôles de refus de la requête.

## save

```python
model.save(path) -> str
```

Écrit un checkpoint LibreYOLO conforme au schéma v1.0\u00a0: state dict et
métadonnées décrites dans le
[schéma des checkpoints](/docs/reference/checkpoint-schema). Un modèle
quantifié contient en plus son manifeste `quant`, afin que `LibreYOLO(path)`
restaure la structure quantifiée et ses échelles.

## quantize, quant_info et dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Quantifie sur place et renvoie le modèle. `recipe` est l'une des conversions
`fp16` et `bf16`, des recettes Conv et Linear `int8` et `fp8`, ou des recettes
réservées à Linear `w4a16`, `w4a8`, `nvfp4`, `mxfp4` et `int2`, prises en
charge par des familles transformer comme RF-DETR. `int2` nécessite QAT.
`calib` accepte le chemin d'un data.yaml ou le nom d'un dataset intégré et lit
les images uniquement en forward, sans jamais lire les étiquettes. Transmettez
`calib=None` pour omettre la calibration. `algorithm` vaut `"minmax"`,
`"percentile"` ou `"auto"`.

`model.quant_info()` renvoie le résumé de l'état de quantification, ou `None`
pour un modèle flottant. `model.dequantize()` restaure les modules flottants
sur place tout en conservant les poids maîtres entraînés avec la
quantification. C'est le pont entre QAT et
`export(format="onnx", int8=True, data=...)`.

## info et layers

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` renvoie un dictionnaire compatible JSON et consigne un résumé lisible
lorsque `verbose` vaut true. `get_available_layer_names` énumère les couches
qu'une configuration de distillation ou d'extraction de caractéristiques peut
nommer.

## Graphes CUDA

Disponibles sur les familles dont l'attribut de classe `SUPPORTS_CUDA_GRAPH`
vaut true. Le rejeu est identique bit à bit à l'exécution eager.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # context manager
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Un graphe capturé n'est valide que pour la forme exacte de sa capture. `batch`
et `imgsz` doivent donc correspondre à l'appel `predict` ultérieur.
`capture_graph` retire le coût de capture de la première requête. `mode`
accepte `True` ou `"on"` pour capturer à la première utilisation, `"auto"`
pour attendre qu'une forme se répète et `False` pour ne rien faire.
`capture_graph` lève `NotImplementedError` si la famille n'a pas activé la
fonctionnalité, et `CudaGraphUnavailable` si la capture échoue.

## Appareil et dtype

Les objets `Results` possèdent `.to()`, `.cpu()`, `.cuda()` et `.numpy()`.
Consultez les [types de résultats](/docs/reference/results-types). Le modèle
lui-même est déplacé en transmettant `device=` à `predict` ou lors de sa
construction.
