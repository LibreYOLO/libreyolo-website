---
title: EfficientDet
families:
  - efficientdet
seo_title: "EfficientDet\_: la détection d'objets dans LibreYOLO"
description: "Exécutez EfficientDet D0-D4 dans LibreYOLO\_: des détecteurs BiFPN pour la prédiction, la validation et l'export vers ONNX, TensorRT et OpenVINO sous licence Apache-2.0."
lead: >-
  EfficientDet associe un backbone EfficientNet à un réseau pyramidal de
  caractéristiques bidirectionnel répété (BiFPN) et met à l'échelle
  conjointement la profondeur, la largeur et la résolution sur cinq tailles.
  LibreYOLO le fournit comme détecteur en inférence seule.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - détection d'objets python
  - compound scaling
  - détecteur léger onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Installation

EfficientDet ne demande aucun extra optionnel. Tout ce qu'il importe fait partie
de l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoient toutes les familles, si bien
que passer à un autre détecteur ne demande qu'une ligne de changement.
EfficientDet décode des candidats à base d'ancres puis applique une
non-maximum suppression classe par classe, si bien que `conf`, `iou` et
`max_det` ont tous un effet réel ici. Voir [la prédiction](/docs/predict) pour
les sources, le streaming et le traitement des résultats.

## Variantes

Cinq tailles, de D0 à D4. Chaque palier associe un backbone EfficientNet plus
grand à un BiFPN plus profond et plus large ainsi qu'à une tête de prédiction
plus profonde, si bien que le nombre de paramètres et le calcul augmentent de
concert, selon la règle de compound scaling de l'article.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurées sur n'importe quel dataset dans le
format avec lequel vous avez entraîné.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` d'après l'extension de son
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Les checkpoints D0-D4 de LibreYOLO sont convertis via le projet
rwightman/efficientdet-pytorch, sous licence Apache-2.0, qui reflète lui-même
les poids officiels entraînés sous TensorFlow de google/automl sans modifier les
tenseurs appris. Aucun code source du projet
zylo117/Yet-Another-EfficientDet-Pytorch, sous licence LGPL, n'a été consulté ni
utilisé.

</provenance-box>
