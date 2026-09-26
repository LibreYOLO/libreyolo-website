---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet dans LibreYOLO : prédire, valider et exporter'
description: >-
  Exécutez RetinaNet dans LibreYOLO pour la détection d'objets à une étape avec
  focal loss. Installez, prédisez, validez et exportez le portage torchvision
  BSD-3-Clause.
lead: >-
  RetinaNet est un détecteur à une étape entraîné avec la focal loss, qui réduit
  le poids des négatifs faciles afin qu'une grille dense d'ancres conserve une
  bonne exactitude sans étape distincte de proposition. LibreYOLO porte
  l'implémentation torchvision pour la détection.
keywords:
  - RetinaNet
  - focal loss
  - détection d'objets
  - détecteur à une étape
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreRetinaNetr50v2.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Installer

RetinaNet ne nécessite aucun extra facultatif. Tous ses imports figurent dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est le même pour toutes les familles, remplacer le
détecteur ne demande donc de modifier qu'une ligne. `conf` et `iou` définissent
les seuils de confiance et de NMS ; RetinaNet conserve son étape NMS amont sur
la grille dense d'ancres. Consultez la [prédiction](/docs/predict) pour les
sources, le streaming et le traitement des résultats.

## Variantes

Deux tailles sont proposées, toutes deux avec ResNet-50 et une pyramide de
caractéristiques : `r50` utilise la tête originale, tandis que `r50v2` la
remplace par une tête GroupNorm et un bloc P6 plus large alimenté par la dernière
étape du backbone au lieu de la sortie FPN.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset dans le format
utilisé pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

RetinaNet s'exporte uniquement vers ONNX, avec un batch de 1. RetinaNet
redimensionne les images vers une entrée variable qui conserve les proportions.
LibreYOLO force donc `dynamic=True` quelle que soit la valeur fournie, afin que
le graphe reste valide pour des sources de formes différentes. Un fichier
`.onnx` exporté se recharge par `LibreYOLO()` à partir de son suffixe de fichier
et renvoie le même objet `Results`.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>
