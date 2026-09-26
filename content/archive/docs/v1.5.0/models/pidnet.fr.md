---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet : prédire et exporter une segmentation temps réel sous MIT'
description: >-
  Utilisez PIDNet dans LibreYOLO pour la segmentation sémantique en temps réel.
  Installez, prédisez, validez et exportez les checkpoints Cityscapes s/m/l sous
  MIT.
lead: >-
  Un réseau de segmentation sémantique à trois branches qui ajoute une branche
  dédiée aux contours à une conception inspirée d'un correcteur
  proportionnel-intégral-dérivé, avec l'inférence en temps réel comme objectif.
  LibreYOLO le fournit uniquement pour la segmentation sémantique.
keywords:
  - PIDNet
  - segmentation sémantique temps réel
  - segmentation sensible aux contours
  - Cityscapes
  - prédiction dense
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # identifiants de classe (H, W)
        print(mask.classes)      # identifiants triés présents dans l'image
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibrePIDNets-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Installer

PIDNet ne nécessite aucun extra facultatif. Tous ses imports figurent dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement. Le suffixe de nom de fichier `-sem` est obligatoire pour
cette famille.

<code-tabs name="predict" />

La segmentation sémantique renvoie un identifiant de classe par pixel, et non
des boîtes. `result.semantic_mask` contient donc un tableau `(H, W)` dans
`.data` et la liste des identifiants de classe présents dans l'image dans
`.classes`. `conf`, `iou` et `max_det` sont acceptés pour la parité de l'API,
mais n'ont aucun effet : le modèle affecte une classe à chaque pixel par argmax,
sans seuil de confiance ni étape NMS. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et le traitement des
résultats.

## Variantes

Trois tailles sont proposées, toutes avec une entrée fixe de 1024 px. Les
checkpoints publiés sont des conversions des poids PIDNet Cityscapes officiels,
avec 19 classes.

LibreYOLO n'entraîne pas PIDNet : `train()` lève `NotImplementedError` pour cette
famille, que le [niveau de prise en charge](/docs/models) ci-dessus indique comme
réservée à l'inférence.

## Valider

`val()` renvoie `metrics/mIoU` et `metrics/pixel_accuracy`, mesurés sur tout
dataset dans le format utilisé pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge par `LibreYOLO()` à partir de son suffixe de
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint et
renvoie le même objet `Results`. La page [Export](/docs/export) répertorie les
arguments acceptés par chaque format.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
