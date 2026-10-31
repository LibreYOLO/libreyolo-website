---
title: EoMT
families:
  - eomt
seo_title: "EoMT\_: prédire la segmentation sémantique, d'instances et panoptique"
description: >-
  Utilisez EoMT dans LibreYOLO pour la segmentation sémantique, d'instances et
  panoptique sur un vision transformer DINOv2 standard, sans decoder. Sous
  licence MIT.
lead: "Un réseau de segmentation bâti sur un vision transformer standard, sans pixel decoder dédié\_: ce sont des queries apprises ajoutées à l'encodeur lui-même qui prédisent les masques. LibreYOLO le prend en charge pour la segmentation sémantique, d'instances et panoptique."
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - segmentation panoptique python
  - segmentation d'instances python
  - segmentation sémantique vision transformer
last_verified: 1.5.0
snippets:
  predict:
    - label: Sémantique
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe
        print(mask.classes)      # ids de classe présents dans l'image, triés
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -seg dans le nom de fichier sélectionne la tâche instance,
        # donc aucun argument task n'est nécessaire ici.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Panoptique
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) ids de segment
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Sémantique
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masques
        print(metrics["metrics/mAP50-95(B)"])   # boîtes
    - label: Panoptique
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Installation

EoMT ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement. Le suffixe de tâche dans le nom de fichier (`-sem`, `-seg`,
`-panoptic`) sélectionne la tâche, et `LibreYOLO()` la déduit de ce nom de
fichier, donc aucun argument `task=` n'est nécessaire.

<code-tabs name="predict" />

La segmentation sémantique remplit `result.semantic_mask`, un tableau `(H, W)`
d'ids de classe sur `.data`. La segmentation d'instances remplit `result.boxes`
et `result.masks`, la même forme que renvoie toute autre famille de
segmentation. La segmentation panoptique remplit `result.panoptic` : une
carte d'ids de segment `(H, W)` sur `.data`, plus `.segments_info`, une liste de
dicts `{"id", "category_id"}`, un par segment. `conf` filtre la sélection des
queries ; `iou` n'a aucun effet sur la tâche sémantique, puisqu'elle fait
un argmax par pixel sans étape de NMS. Voir [la prédiction](/docs/predict) pour
les sources, le streaming et le traitement des résultats.

## Variantes

Trois tailles d'encodeur, s/b/l, toutes adossées à DINOv2. Le checkpoint
sémantique est entraîné sur ADE20K en 512 px ; les checkpoints
instance et panoptique sont entraînés sur COCO en 640 px, avec un second
checkpoint instance entraîné en 1280 px. En amont, les poids de
segmentation d'instances DINOv2 ne sont fournis qu'en taille l ; s et b ne
sont publiés que pour le sémantique et le panoptique. Des variantes d'EoMT
adossées à DINOv3 existent en amont mais ne sont pas fournies ici, car elles
dépendent de poids DINOv3 non commerciaux à accès restreint.

LibreYOLO n'entraîne pas EoMT : `train()` lève `NotImplementedError` pour
cette famille, ce que le [niveau de support](/docs/models) ci-dessus signale
comme inférence seule.

## Valider

`val()` aiguille selon la tâche. Le sémantique renvoie `metrics/mIoU` et
`metrics/pixel_accuracy`. La segmentation d'instances renvoie les mêmes clés de
mAP masques et boîtes que les autres familles de segmentation. Le panoptique
renvoie la Panoptic Quality sous `metrics/PQ`, décomposée en `metrics/SQ`
(qualité de segmentation) et `metrics/RQ` (qualité de reconnaissance), plus
`metrics/PQ_things` et `metrics/PQ_stuff`.

<code-tabs name="val" />

## Exporter

<export-matrix />

Seule la tâche sémantique s'exporte aujourd'hui : la segmentation
d'instances et la segmentation panoptique appellent `export()` et obtiennent
`NotImplementedError`, parce que leur sortie de masques par query n'a pas encore
de contrat d'export runtime. Un artefact sémantique exporté se recharge via
`LibreYOLO()` selon son extension de fichier, si bien qu'un fichier `.onnx` ou
`.engine` se comporte comme un checkpoint et renvoie le même `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
