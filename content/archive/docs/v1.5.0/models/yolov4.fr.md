---
title: YOLOv4
families:
  - yolo4
seo_title: "YOLOv4\_: exécuter, valider et exporter dans LibreYOLO"
description: "Exécutez YOLOv4 dans LibreYOLO\_: une famille historique figée, réservée à l'inférence, avec un backbone CSPDarknet-53. Prédisez, validez et exportez sous une licence du domaine public."
lead: >-
  YOLOv4 associe un backbone CSPDarknet-53, un bloc SPP et un neck PANet à des
  activations Mish. LibreYOLO le conserve comme pièce historique figée, réservée
  à l'inférence, dans les tailles tiny et base.
keywords:
  - YOLOv4
  - Darknet
  - CSPDarknet-53
  - PANet
  - détection d'objets
  - activation Mish
  - ancien modèle YOLO
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO4b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreYOLO4b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 6070bb4a09d75416
---

## Installer

YOLOv4 ne nécessite aucun extra au-delà du package de base.

```bash
pip install libreyolo
```

## Prédire

Cette famille est réservée à l'inférence\u00a0: `train()` lève
`NotImplementedError`, cette page ne comporte donc aucune section Entraîner.
La prédiction, la validation et l'export sont tous pris en charge. Les poids
sont téléchargés depuis Hugging Face à la première utilisation et mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui de toutes les familles, le remplacement du
détecteur ne demande donc de modifier qu'une ligne. `conf` filtre selon le
seuil de confiance et `iou` selon le seuil de NMS, appliqués après la mise à
l'échelle centrale `scale_x_y` propre à chaque tête. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset au format utilisé
pour la validation.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. Vous pouvez également exécuter le graphe dans
un runtime nu, sans installer LibreYOLO, mais vous devez alors écrire vous-même
le prétraitement et le post-traitement.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />

