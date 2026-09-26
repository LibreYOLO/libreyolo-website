---
title: DeepLabv3
families:
  - deeplabv3
seo_title: "DeepLabv3\_: prédire et exporter la segmentation sémantique ASPP"
description: >-
  Utilisez DeepLabv3 dans LibreYOLO pour la segmentation sémantique. Installez,
  prédisez, validez et exportez les checkpoints ResNet et MobileNetV3 de
  torchvision.
lead: >-
  Un réseau de segmentation sémantique qui agrège les caractéristiques à
  plusieurs taux de dilatation en parallèle (atrous spatial pyramid pooling)
  avant de classer chaque pixel. LibreYOLO le fournit uniquement pour la
  segmentation sémantique.
keywords:
  - DeepLabv3
  - ASPP
  - atrous spatial pyramid pooling
  - segmentation sémantique python
  - segmenter une image pixel par pixel
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de classe
        print(mask.classes)      # ids de classe présents dans l'image, triés
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Installation

DeepLabv3 ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement. Le suffixe `-sem` dans le nom de fichier est obligatoire pour
cette famille.

<code-tabs name="predict" />

La segmentation sémantique renvoie un id de classe par pixel, pas des bounding
boxes, si bien que `result.semantic_mask` porte un tableau `(H, W)` sur `.data`
et la liste des ids de classe présents dans l'image sur `.classes`. `conf`,
`iou` et `max_det` sont acceptés par parité d'API mais n'ont aucun effet : le
modèle attribue une classe à chaque pixel par argmax, sans seuil de confiance ni
étape de NMS. Voir [la prédiction](/docs/predict) pour les sources, le streaming
et le traitement des résultats.

## Variantes

Trois backbones : ResNet-50 dilaté, ResNet-101 dilaté et MobileNetV3-Large
dilaté. Il s'agit de DeepLabv3, pas de DeepLabv3+, donc il n'y a ni étape de
decoder ni raffinement par CRF, ce qui suit l'implémentation de torchvision
plutôt que le code de référence du papier lui-même.

LibreYOLO n'entraîne pas DeepLabv3 : `train()` lève `NotImplementedError` pour
cette famille, ce que le [niveau de support](/docs/models) ci-dessus signale
comme inférence seule. Les trois checkpoints publiés sont les poids torchvision
entraînés sur COCO avec les étiquettes de VOC, convertis pour le chargeur de
LibreYOLO.

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
