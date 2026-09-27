---
title: FCN
families:
  - fcn
seo_title: "FCN\_: prédire et exporter un FCN ResNet sous BSD-3-Clause"
description: >-
  Utilisez FCN dans LibreYOLO pour la segmentation sémantique. Installez,
  prédisez, validez et exportez les checkpoints FCN à ResNet dilaté de
  torchvision.
lead: >-
  Un classifieur dense pixel par pixel qui remplace les couches entièrement
  connectées d'un détecteur par des convolutions, si bien qu'il produit une
  carte de classes en pleine résolution au lieu de boîtes. LibreYOLO le fournit
  uniquement pour la segmentation sémantique.
keywords:
  - FCN
  - fully convolutional network
  - segmentation sémantique python
  - prédiction dense pixel par pixel
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe
        print(mask.classes)      # ids de classe présents dans l'image, triés
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Installation

FCN ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

La segmentation sémantique renvoie un id de classe par pixel, pas des boîtes, si
bien que `result.semantic_mask` porte un tableau `(H, W)` sur `.data` et la
liste des ids de classe présents dans l'image sur `.classes`. `conf`, `iou` et
`max_det` sont acceptés par parité d'API mais n'ont aucun effet : le modèle
attribue une classe à chaque pixel par argmax, sans seuil de confiance ni étape
de NMS. Voir [la prédiction](/docs/predict) pour les sources, le streaming et le
traitement des résultats.

## Variantes

Deux profondeurs de ResNet, toutes deux avec une entrée fixe de 520 px. Le
graphe d'inférence de la bibliothèque est le FCN à ResNet dilaté de torchvision,
pas le réseau FCN-8s à base de VGG du papier d'origine, avec ses skip
connections.

LibreYOLO n'entraîne pas FCN : `train()` lève `NotImplementedError` pour cette
famille, ce que le [niveau de support](/docs/models) ci-dessus signale comme
inférence seule. Les deux checkpoints publiés sont les poids torchvision
entraînés sur COCO, convertis pour le chargeur de LibreYOLO.

## Valider

`val()` renvoie `metrics/mIoU` et `metrics/pixel_accuracy`, mesurés sur
n'importe quel dataset au format sur lequel vous avez entraîné.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. [L'export](/docs/export) liste les
arguments que chaque format accepte.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>
